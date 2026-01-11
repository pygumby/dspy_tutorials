import getpass
import os
import random
import tarfile

import bm25s
import dspy
import numpy
import Stemmer
import ujson
from dotenv import load_dotenv
from dspy.datasets import DataLoader
from dspy.utils import download

load_dotenv()
if not os.environ.get("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together: ")
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llama8b = dspy.LM(
    "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    max_tokens=3000,
)
gpt4o = dspy.LM("openai/gpt-4o", max_tokens=3000)

dspy.configure(lm=llama8b)

folder_name = "multihop_search"
file_name = "wiki.abstracts.2017"

download(f"https://huggingface.co/dspy/cache/resolve/main/{file_name}.tar.gz")
os.rename(f"{file_name}.tar.gz", f"{folder_name}/{file_name}.tar.gz")

with tarfile.open(f"{folder_name}/{file_name}.tar.gz", "r:gz") as tar:
    tar.extractall(path=folder_name, filter="data")

corpus: list[str] = []

with open(f"{folder_name}/{file_name}.jsonl") as file:
    for line in file:
        obj: dict[str, list[str]] = ujson.loads(line)
        corpus.append(f"{obj['title']} | {' '.join(obj['text'])}")

print(f"Corpus size: {len(corpus)}")

stemmer = Stemmer.Stemmer("english")
corpus_tokens: bm25s.tokenization.Tokenized = bm25s.tokenize(
    corpus,
    stopwords="en",
    stemmer=stemmer,
)

# print(f"Tokenized corpus size: {len(corpus_tokens.ids)}")
# print(f"Vocabulary size: {len(corpus_tokens.vocab)}")
# print(f"First document's token IDs: {corpus_tokens.ids[1][:10]}")
# print(f"Sample vocabulary entries: {list(corpus_tokens.vocab.items())[:30]}")

retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)

kwargs = dict(
    fields=("claim", "supporting_facts", "hpqa_id", "num_hops"),
    input_keys=("claim",),
)
hover = DataLoader().from_huggingface(
    dataset_name="vincentkoc/hover-parquet",
    split="train",
    trust_remote_code=True,
    **kwargs,
)

hpqa_ids = set()
hover = [
    dspy.Example(
        claim=x.claim,
        titles=list(set([y["key"] for y in x.supporting_facts])),
    ).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3
    and x["hpqa_id"] not in hpqa_ids
    and not hpqa_ids.add(x["hpqa_id"])  # type: ignore[func-returns-value]
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:50], hover[50:500], hover[650:]
# Original: trainset = hover[:200]

# example = trainset[0]

# print(f"Claim: {example.claim}")
# print(f"Pages that must be retrieved: {example.titles}")


def search(query: str, k: int) -> dict[str, float]:
    tokens: bm25s.tokenization.Tokenized = bm25s.tokenize(
        query, stopwords="en", stemmer=stemmer, show_progress=False
    )
    results: tuple[numpy.ndarray, numpy.ndarray] = retriever.retrieve(
        tokens, k=k, n_threads=1, show_progress=False
    )
    docs, scores = results
    run: dict[str, float] = {
        corpus[doc]: float(score) for doc, score in zip(docs[0], scores[0])
    }
    return run


class Hop(dspy.Module):
    def __init__(self, num_docs: int = 10, num_hops: int = 3) -> None:
        # Original: num_hops = 4
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought("claim, notes -> query")
        self.append_notes = dspy.ChainOfThought(
            "claim, notes, context -> new_notes: list[str], titles: list[str]"
        )

    def forward(self, claim: str) -> dspy.Prediction:
        notes: list[str] = []
        titles: list[str] = []

        for i in range(self.num_hops):
            query: str = self.generate_query(claim=claim, notes=notes).query
            context: dict[str, float] = search(query, self.num_docs)
            prediction: dspy.Prediction = self.append_notes(
                claim=claim, notes=notes, context=context
            )
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)

            # print(f"Iteration: {i}")
            # print(f"Query: {query}")
            # print(f"Context: {context}")
            # print(f"Notes: {notes}")
            # print(f"Titles: {titles}")
            # print("\n")

        return dspy.Prediction(notes=notes, titles=list(set(titles)))


# hop = Hop()

# print(hop(claim="Harry Potter was born in Hogwarts."))

# hop.save("multihop_search/multihop_search.json")


def top_5_recall(
    example: dspy.Example, pred: dspy.Prediction, trace: bool | None = None
) -> float | bool:
    gold_titles = example.titles
    recall: float = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)

    # If we're "bootstrapping" for optimization, return True if and only if the recall
    # is perfect
    if trace is not None:
        return recall >= 1.0

    # If we're just doing inference, just measure the recall
    return recall


evaluate = dspy.Evaluate(
    devset=devset,
    metric=top_5_recall,
    num_threads=16,
    display_progress=True,
    display_table=5,
)

# evaluate(hop)  # Average Metric: 105.33 / 450 (23.4%)

optimizer_kwargs = dict(prompt_model=gpt4o, teacher_settings=dict(lm=gpt4o))
optimizer = dspy.MIPROv2(
    metric=top_5_recall,
    auto="light",  # Original: auto="medium"
    num_threads=16,
    **optimizer_kwargs,
)

optimization_kwargs = dict(
    minibatch_size=20,  # Original: minibatch_size=40
    minibatch_full_eval_steps=4,  # Original: minibatch_full_eval_steps=8
)
# optimized_hop = optimizer.compile(
#     hop,
#     trainset=trainset,
#     max_bootstrapped_demos=2,  # Original: max_bootstrapped_demos=4
#     max_labeled_demos=2,  # Original: max_labeled_demos=4
#     **optimization_kwargs,
# )

# evaluate(optimized_hop)  # Average Metric: 256.33 / 450 (57.0%)

# print(
#     optimized_hop(
#         claim=(
#             "The author of the 1960s unproduced script written for The Beatles, Up"
#             "Against It, and Bernard-Marie Koltès are both playwrights."
#         )
#     ).titles
# )
# dspy.inspect_history(n=2)

# optimized_hop.save("multihop_search/multihop_search_optimized.json")

loaded_hop = Hop()
loaded_hop.load("multihop_search/multihop_search_optimized.json")

print(
    loaded_hop(
        claim=(
            "The author of the 1960s unproduced script written for The Beatles, Up"
            "Against It, and Bernard-Marie Koltès are both playwrights."
        )
    ).titles
)
