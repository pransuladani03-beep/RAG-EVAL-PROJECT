# eval_retriever.py
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of execution path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Increase timeout so local model evaluation doesn't get interrupted
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import LocalModel
from deepeval.evaluate.configs import AsyncConfig

from src.reranker import RerankingRetriever
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL_NAME = "Qwen2.5-7B-Instruct"
THRESHOLD = 0.8

# Initialize Qwen2.5-7B-Instruct as judge
HF_TOKEN = os.getenv("HF_TOKEN")

# If a valid Hugging Face token (starting with 'hf_') is present, use Hugging Face router;
# otherwise seamlessly connect to local Qwen2.5-7B-Instruct running via Ollama.
if HF_TOKEN and HF_TOKEN.startswith("hf_"):
    judge_model = LocalModel(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
else:
    judge_model = LocalModel(
        model=JUDGE_MODEL_NAME,
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
    )


def run(retriever):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge_model, async_mode=False, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge_model, async_mode=False, include_reason=True),
    ]

    # 4. EVALUATE --- evaluate each metric cleanly, sequential for local model stability
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 3,
            "judge_model": JUDGE_MODEL_NAME,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(RerankingRetriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())
