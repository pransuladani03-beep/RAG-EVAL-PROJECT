# eval_rag_pipeline.py
import os
import re
import json
import time
from dotenv import load_dotenv

import deepeval.models.llms.utils as deutils
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline, normalize_query_for_retrieval
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

# Auto-repair unescaped backslashes from LLM so json.loads does not fail
_orig_trim = deutils.trim_and_load_json
def _safe_trim(s):
    try:
        return _orig_trim(s)
    except Exception:
        clean = re.sub(r'\\(?![/\"\\bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', s)
        start, end = clean.find("{"), clean.rfind("}") + 1
        return json.loads(clean[start:end], strict=False)

deutils.trim_and_load_json = _safe_trim


class PacedNvidiaModel(LocalModel):
    """Paces judge requests and uses exponential backoff on 429 rate limits."""
    def generate(self, *args, **kwargs):
        time.sleep(10)  # Pause 10s before each judge request (~6 requests/min)
        for attempt in range(5):
            try:
                return super().generate(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower() or "too many requests" in str(e).lower():
                    wait = 20 * (attempt + 1)
                    print(f"\n[Rate Limit 429] Judge rate limit hit. Waiting {wait}s before retry ({attempt + 1}/5)...")
                    time.sleep(wait)
                else:
                    raise
        return super().generate(*args, **kwargs)


GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
JUDGE_MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"
THRESHOLD = 0.8

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if NVIDIA_API_KEY:
    judge_model = PacedNvidiaModel(
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


def run(rag):
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        time.sleep(10)  # Request delay: pauses 10s between queries to stay under NVIDIA RPM limit
        for attempt in range(5):
            try:
                result = rag.invoke(g["query"])          # retrieve -> rerank -> generate
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower() or "too many requests" in str(e).lower():
                    wait = 20 * (attempt + 1)
                    print(f"\n[Rate Limit 429] Generator rate limit hit. Waiting {wait}s before retry ({attempt + 1}/5)...")
                    time.sleep(wait)
                else:
                    raise

        test_cases.append(
            LLMTestCase(
                input=normalize_query_for_retrieval(g["query"]),
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge_model, include_reason=True, async_mode=False),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge_model, include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge_model, include_reason=True, async_mode=False),
    ]

    # 4. EVALUATE
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=False),
    )

    # 5. PRINT EXACT FAILURE DIAGNOSTICS
    for tr in result.test_results:
        if not tr.success:
            print(f"\n❌ FAILED QUERY: {tr.input}")
            for m in tr.metrics_data:
                if not m.success:
                    print(f"   - {m.name}: {m.score:.2f} | Reason: {m.reason}")

    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
