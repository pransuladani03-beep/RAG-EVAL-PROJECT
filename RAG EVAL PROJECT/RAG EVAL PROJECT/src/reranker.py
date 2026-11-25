from sentence_transformers import CrossEncoder
try:
    from src.retriever import build_retriever
except ModuleNotFoundError:
    from retriever import build_retriever


class RerankingRetriever:
    def __init__(self, top_k: int = 3, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.base_retriever = build_retriever()
        self.reranker = CrossEncoder(model_name)
        self.top_k = top_k

    def invoke(self, query: str):
        # 1. Retrieve candidate documents from the base retriever
        docs = self.base_retriever.invoke(query)
        if not docs:
            return []

        # 2. Score query-document pairs using the CrossEncoder
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)

        # 3. Sort documents by score descending and take top_k
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[: self.top_k]]


if __name__ == "__main__":
    retriever = RerankingRetriever()
    results = retriever.invoke("what is regression testing?")
    for r in results:
        print(f"[Session {r.metadata.get('session', '?')}] {r.page_content[:150]}...\n")
