import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Same model used in embeddings_pipeline.py
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def build_faiss_index(embeddings):

    if len(embeddings) == 0:
        return None

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(
            embeddings,
            dtype=np.float32
        )
    )

    return index


def search_news(
    query,
    index,
    chunks,
    top_k=5
):

    if (
        index is None
        or len(chunks) == 0
    ):
        return []

    query_embedding = embedding_model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(
            query_embedding,
            dtype=np.float32
        ),
        top_k
    )

    results = []

    for idx in indices[0]:

        if idx < len(chunks):

            results.append(
                chunks[idx].page_content
            )

    return results