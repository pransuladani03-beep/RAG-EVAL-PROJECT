"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m evals.eval_generator
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Disable artificial timeout watchdog so cloud model latency does not abort the run
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "true"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import LocalModel

try:
    from src.rag_eval_project.generator import generate
except ModuleNotFoundError:
    from src.generator import generate
from evals.harness import load_goldens, summarize_by_metric, print_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"
JUDGE_MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"
THRESHOLD = 0.8

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if NVIDIA_API_KEY:
    judge_model = LocalModel(
        model=JUDGE_MODEL_NAME,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        generation_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
else:
    judge_model = LocalModel(
        model=JUDGE_MODEL_NAME,
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
    )


def run():
    # 1. LOAD the faithfulness golden set (query + ideal_context)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for g in goldens:
        context = g["ideal_context"]              # known-good context (list of chunk strings)
        answer = generate(g["query"], context)    # RUN the generator -> actual_output

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,             # the generated answer we're judging
                retrieval_context=context,        # faithfulness checks the answer against THIS
                # no expected_output --- faithfulness never reads it
            )
        )

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
            async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge_model,
            include_reason=True,
            async_mode=False,
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=True),
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())
