import getpass
import os
import random

import dspy
import ujson
from dotenv import load_dotenv
from dspy.evaluate import SemanticF1

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

########################################################################################
# Configuring the DSPy environment                                                     #
########################################################################################

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

########################################################################################
# Exploring some basic DSPy Modules                                                    #
########################################################################################

qa = dspy.Predict("question: str -> response: str")
# print(qa(question='What are "high memory" and "low memory" in Linux?'))
# print(dspy.inspect_history(n=1))

cot = dspy.ChainOfThought("question -> response")  # `str` omitted, as it's the default
# print(cot(question="Should curly braces appear on their own line?"))

########################################################################################
# Manipulating Examples in DSPy                                                        #
########################################################################################

with open("rag/ragqa_arena_tech_examples.jsonl") as file:
    data: list[str] = [ujson.loads(line) for line in file]
# print(data[0])

examples: list[dspy.Example] = [dspy.Example(**d).with_inputs("question") for d in data]

# Let's pick one example from the data
example: dspy.Example = examples[2]
# print(example)

random.Random(0).shuffle(examples)
trainset: list[dspy.Example] = examples[:200]
devset: list[dspy.Example] = examples[200:500]
testset: list[dspy.Example] = examples[500:1000]
# print(f"{len(trainset)}, {len(devset)}, {len(testset)}")

########################################################################################
# Evaluation in DSPy                                                                   #
########################################################################################

# Instantiate the metric
metric: dspy.Module = SemanticF1(decompositional=True)

# Produce a prediction from out `cot` module, using the `example` above as input
pred: dspy.Prediction = cot(**example.inputs())

# Compute the metric score for the prediction
score: dspy.Prediction = metric(example, pred)

# print(f"Question:\n{example.question}\n")
# print(f"Gold response:\n{example.response}\n")
# print(f"Predicted response:\n{pred.response}\n")
# print(f"Semantic F1 score:\n{score:.2f}")

# The final DSPy module call above actually happens inside `metric`
# dspy.inspect_history(n=1)

# Define an evaluator that we can re-use
evaluate = dspy.Evaluate(
    devset=devset,
    metric=metric,
    num_threads=24,
    display_progress=True,
    display_table=2,
)

# evaluate(cot)

########################################################################################
# Basic Retrieval-Augmented Generation (RAG)                                           #
########################################################################################

max_characters = 6000  # Truncate >99th percentile of documents
topk_docs_to_retrieve = 5  # Number of documents to retrieve per search query

with open("rag/ragqa_arena_tech_corpus.jsonl") as file:
    corpus: list[str] = [ujson.loads(line)["text"][:max_characters] for line in file]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")

embedder = dspy.Embedder("openai/text-embedding-3-small", dimensions=512)

# Calls the OpenAI API to embed the whole corpus and stores these embeddings locally
search = dspy.retrievers.Embeddings(
    embedder=embedder,
    corpus=corpus,
    k=topk_docs_to_retrieve,
)


class RAG(dspy.Module):
    def __init__(self) -> None:
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question: str) -> dspy.Prediction:
        context: list[str] = search(question).passages
        return self.respond(context=context, question=question)


rag = RAG()
# print(rag(question='What are "high memory" and "low memory" in Linux?'))
# dspy.inspect_history()

# evaluate(rag)

########################################################################################
# Using a DSPy Optimizer to improve your RAG prompt                                    #
########################################################################################

optimizer = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)

# The run below has a cost around $1.5 (for the medium auto setting) and may take some
# 20-30 minutes depending on your number of threads
# optimized_rag = optimizer.compile(
#     RAG(),
#     trainset=trainset,
#     max_bootstrapped_demos=2,
#     max_labeled_demos=2,
# )

# baseline = rag(question="cmd+tab does not work on hidden or minimized windows")
# print(f"Baseline response: {baseline.response}")

# pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
# print(f"Optimized response: {pred.response}")

# dspy.inspect_history(n=2)

# evaluate(optimized_rag)

########################################################################################
# Keeping an eye on cost                                                               #
########################################################################################

# In USD, as calculated by LiteLLM for certain providers
cost = sum([x["cost"] for x in lm.history if x["cost"] is not None])
# print(f"Cost: USD {cost}")

########################################################################################
# Saving and loading                                                                   #
########################################################################################

# optimized_rag.save("rag/optimized_rag.json")

# loaded_rag = RAG()
# loaded_rag.load("rag/optimized_rag.json")

# print(loaded_rag(question="cmd+tab does not work on hidden or minimized windows"))
