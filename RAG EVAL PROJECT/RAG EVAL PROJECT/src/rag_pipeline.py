import inspect

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src.reranker import RerankingRetriever
from src.generator import generate


def normalize_query_for_retrieval(query: str) -> str:
    """Converts Hinglish question words into English so dense embeddings match English transcripts."""
    replacements = {
        "kya hota hai": "what is",
        "kya hai": "what is",
        "kaise": "how",
        "kyu": "why",
        "kyun": "why",
        "aur": "and",
    }
    lower_q = query.strip().lower()
    for hi, en in replacements.items():
        lower_q = lower_q.replace(hi, en)
    return lower_q


class RagPipeline:
    def __init__(self, fetch_k=17, top_k=3, min_score=0.39):
        # Safely initialize RerankingRetriever with only the keyword arguments it accepts
        sig_params = inspect.signature(RerankingRetriever.__init__).parameters
        kwargs = {}
        if "fetch_k" in sig_params:
            kwargs["fetch_k"] = fetch_k
        if "top_k" in sig_params:
            kwargs["top_k"] = top_k
        elif "k" in sig_params:
            kwargs["k"] = top_k

        self.retriever = RerankingRetriever(**kwargs)
        self.top_k = top_k
        self.min_score = min_score

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        # 1. NORMALIZE: convert Hinglish terms so English embeddings match accurately
        search_query = normalize_query_for_retrieval(query)

        # 2. RETRIEVE: over-fetch then rerank down to top_k Documents
        docs = self.retriever.invoke(search_query)

        # 3. FILTER: drop low-relevance chunks while always keeping at least the top-1 chunk
        filtered_docs = [
            doc for i, doc in enumerate(docs)
            if i == 0 or doc.metadata.get("relevance_score", 1.0) >= self.min_score
        ][:self.top_k]

        # 4. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in filtered_docs]

        # 5. GENERATE: grounded answer from the filtered context
        answer = generate(query, context)

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("what is drift and why does it metter after deployment?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")
