import getpass
import os
import random

import dspy
from dotenv import load_dotenv
from dspy.datasets import DataLoader
from dspy.dsp.utils import dotdict

load_dotenv()
if not os.environ.get("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter API key for Together: ")
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llama3b = dspy.LM("together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo", temperature=0.7)
gpt4o = dspy.LM("openai/gpt-4o", temperature=0.7)

dspy.configure(lm=llama3b)

dataloader_kwargs = dict(
    fields=("claim", "supporting_facts", "hpqa_id", "num_hops"),
    input_keys=("claim",),
)
hover = DataLoader().from_huggingface(
    dataset_name="hover-nlp/hover",
    split="train",
    trust_remote_code=True,
    **dataloader_kwargs,
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
trainset, devset, testset = hover[:100], hover[100:200], hover[650:]

# example = trainset[0]

# print("Claim:", example.claim)
# print("Pages that must be retrieved:", example.titles)

DOCS = {}

# The ColBERT server reference in the tutorial is down:
# url = "http://20.102.90.50:2017/wiki17_abstracts"

# To run a local alternative, do:
# `uv tool install colbert-server`
# `uv tool install ninja`
# `colbert-server serve --from-cache`
url = "http://127.0.0.1:8893/api/search"


def search(query: str, k: int) -> list[str]:
    raw_results: list[dotdict] = dspy.ColBERTv2(url=url)(query, k=k)
    results: list[str] = [x["text"] for x in raw_results]

    for result in results:
        title, text = result.split(" | ", 1)
        DOCS[title] = text

    return results


def search_wikipedia(query: str) -> list[str]:
    """Returns the top-5 results and then the titles of the top-5 to top-30 results"""

    top_k: list[str] = search(query, 30)
    titles: list[str] = [f"`{x.split(' | ')[0]}`" for x in top_k[5:30]]
    top_k: list[str] = top_k[:5]  # type: ignore[no-redef]

    return top_k + [f"Other retrieved pages are have the titles: {', '.join(titles)}"]


def lookup_wikipedia(title: str) -> str:
    """Returns the text of the Wikipedia page, if it exists"""

    if title in DOCS:
        return DOCS[title]

    results = [x for x in search(title, 10) if x.startswith(title + " | ")]

    if not results:
        return f"No Wikipedia page found for title: {title}"

    return results[0]


instructions = "Find all Wikipedia titles relevant to verifying or refuting the claim."
signature = dspy.Signature("claim -> titles: list[str]", instructions)
react = dspy.ReAct(signature, tools=[search_wikipedia, lookup_wikipedia], max_iters=20)

# print(react(claim="David Gregory was born in 1625.").titles[:3])

# react.save("agents.json")


def top_5_recall(
    example: dspy.Example, pred: dspy.Prediction, trace: bool | None = None
) -> float | bool:
    """
    Return the fraction of the gold pages (which are always 3) that are retrieved in the
    top-5 titles returned by the agent
    """
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


def safe_react(claim: str) -> dspy.Prediction:
    try:
        return react(claim=claim)
    except Exception:
        return dspy.Prediction(titles=[])


# Let's evaluate our off-the-shelf agent, Llama 3.2 3B, to see how far we get already
# evaluate(safe_react)  # Average Metric: 25.0 / 100 (25.0%)

optimizer_kwargs = dict(
    teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o, max_errors=999
)

optimizer = dspy.MIPROv2(
    metric=top_5_recall, auto="medium", num_threads=16, **optimizer_kwargs
)

# optimized_react = optimizer.compile(
#     react, trainset=trainset, max_bootstrapped_demos=3, max_labeled_demos=0
# )

# Let's now evaluate again, after optimization
# evaluate(optimized_react)  # Average Metric: 49.33 / 100 (49.3%)

# Next, let's inspect the optimized prompts to understand what it has learned
# print(
#     optimized_react(
#         claim=(
#             "The author of the 1960s unproduced script written for The Beatles, "
#             "Up Against It, and Bernard-Marie Koltès are both playwrights."
#         )
#     ).titles
# )
# dspy.inspect_history(n=2)

# Finally, let's save our optimized program so we can use it again later
# optimized_react.save("agents_optimized.json")

loaded_react = dspy.ReAct(
    "claim -> titles: list[str]",
    tools=[
        search_wikipedia,
        lookup_wikipedia,
    ],
)
loaded_react.load("agents_optimized.json")

print(
    loaded_react(
        claim="The author of the 1960s unproduced script written for The Beatles, "
        "Up Against It, and Bernard-Marie Koltès are both playwrights."
    ).titles
)
