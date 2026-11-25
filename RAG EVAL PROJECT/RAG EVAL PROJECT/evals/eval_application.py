# eval_application.py
import os
import re
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is on sys.path regardless of execution path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Increase timeout and disable watchdog timeouts for evaluation stability
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "true"

import deepeval.models.llms.utils as deutils
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import LocalModel

from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens, summarize_by_metric, print_summary
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA


load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv()

# Auto-repair unescaped backslashes and control characters from LLM so json.loads does not fail
_orig_trim = deutils.trim_and_load_json
def _safe_trim(s):
    try:
        return _orig_trim(s)
    except Exception:
        clean = re.sub(r'\\(?![/\"\\bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', s)
        start, end = clean.find("{"), clean.rfind("}") + 1
        return json.loads(clean[start:end], strict=False)

deutils.trim_and_load_json = _safe_trim

class PacedNvidiaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=0,
            timeout=300,
            model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None):
        time.sleep(10)  # Pace requests (~6 RPM)
        for attempt in range(5):
            try:
                res = self.client.invoke(prompt)
                text = res.content
                if schema is not None:
                    data = _safe_trim(text)
                    return schema(**data)
                return text
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower() or "too many requests" in str(e).lower():
                    print(f"\n[Rate Limit 429] Judge rate limit hit. Waiting 20s (attempt {attempt + 1}/5)...")
                    time.sleep(20)
                else:
                    raise
        return self.client.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return self.model_name

GOLDEN_PATH = "goldens/correctness_goldens.json"
JUDGE_MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"
THRESHOLD = 0.8

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
judge_model = PacedNvidiaJudge(model_name=JUDGE_MODEL_NAME, api_key=NVIDIA_API_KEY)


def run(rag):
    # 1. LOAD queries + ideal answers (test 1 case first to verify judge works!)
    goldens = load_goldens(GOLDEN_PATH)


    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        time.sleep(10)  # Request delay: pauses 10s between queries to stay under NVIDIA RPM limit
        for attempt in range(5):
            try:
                result = rag.invoke(g["question"])          # retrieve -> rerank -> generate
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower() or "too many requests" in str(e).lower():
                    print(f"\n[Rate Limit 429] Generator rate limit hit. Waiting 20s before retry (attempt {attempt + 1}/5)...")
                    time.sleep(20)
                else:
                    raise

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],
                expected_output=g["ideal_answer"],
            )
        )

    # 3. THREE APPLICATION-LEVEL QUALITY METRICS

    # 3a. CORRECTNESS --- reference-based, judges semantic truth
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare the claims in the actual output against the expected output for semantic truth.",
            "The actual output comes from a full course lecture and will naturally contain extra valid examples, components, or synonyms (such as 'workflow level' for 'pipeline level') not written in the short expected output.",
            "Treat all extra course details and valid paraphrases as 100% CORRECT unless they directly and logically contradict the expected output.",
            "Never deduct points in Correctness for missing details (Completeness is graded separately).",
            "If there are no factual errors or direct contradictions, assign a score of 9 or 10.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Factually accurate with zero contradictions; paraphrases and extra lecture details are completely valid."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly accurate with one minor factual error."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or direct contradiction."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge_model,
        strict_mode=False,
        async_mode=False,
    )

 # 3b. COMPLETENESS --- reference-based, judges core concept coverage
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the core concepts in the expected output.",
            "Check if the actual output conveys those same core concepts, even when phrased differently or illustrated with different examples.",
            "Award full credit when the core ideas are covered semantically without requiring exact keyword matches.",
            "Do NOT penalize for extra course context or examples.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Covers all core concepts from the expected output, even if worded differently."),
            Rubric(score_range=(7, 8),  expected_outcome="Covers the main ideas well, missing only a minor detail."),
            Rubric(score_range=(0, 6),  expected_outcome="Misses major core concepts from the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.8,
        model=judge_model,
        strict_mode=False,
        async_mode=False,
    )

    # 3c. STYLE --- reference-free, judges TONE only (note: no EXPECTED_OUTPUT)
    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and tone of the actual output, not whether it is factually correct or complete.",
            "Reward an intuitive, explanatory tone: plain language, the idea explained before any formula or jargon, and technical terms briefly unpacked when used.",
            "Reward a direct, conversational register written in prose, as a CampusX lecture would explain it out loud, rather than a dry, formal, or bullet-list tone.",
            "An analogy or concrete example is a BONUS when the concept is abstract, but a clear, direct, well-explained answer is fully acceptable and must NOT be penalized for not having one.",
            "Penalize answers that are stiff, bureaucratic, structured as a bare list with no explanation, or that use unexplained jargon.",
            "Do NOT reward or penalize based on correctness, completeness, or length --- only on style and tone.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Clearly in a CampusX teaching voice: intuitive, conversational prose that explains before it formalizes."),
            Rubric(score_range=(7, 8),  expected_outcome="Clear, conversational, and well-explained in prose. Fully acceptable even without an analogy or example."),
            Rubric(score_range=(4, 6),  expected_outcome="Understandable but somewhat flat, formal, or list-heavy in places."),
            Rubric(score_range=(0, 3),  expected_outcome="Dry, stiff, bare-list, jargon-heavy, or robotic; does not read like a teaching explanation."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=judge_model,
        strict_mode=False,
        async_mode=False,
    )

    # 4. EVALUATE --- all three together
    result = evaluate(
        test_cases=test_cases,
        metrics=[correctness, completeness, style],
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=True),
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("application", run_local())
