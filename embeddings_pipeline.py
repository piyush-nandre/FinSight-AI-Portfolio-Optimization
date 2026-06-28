from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


# Load embedding model once
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def generate_news_embeddings(news_df):
    """
    Convert news into LangChain documents,
    chunk them, and generate embeddings.
    """

    if news_df.empty:
        return {
            "documents": [],
            "chunks": [],
            "embeddings": [],
            "embedding_dimension": 0
        }

    # Convert news rows into documents
    documents = []

    for _, row in news_df.iterrows():

        content = f"""
        Headline:
        {row['Headline']}

        Source:
        {row['Source']}

        Published:
        {row['Published']}

        Summary:
        {row.get('Summary', '')}

        Article Content:
        {row.get('Content', '')}
        """

        documents.append(
            Document(
                page_content=content
            )
        )

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(
        documents
    )

    # Extract chunk text
    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # Generate embeddings
    embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True
    )

    embedding_dimension = (
        embeddings.shape[1]
        if len(embeddings) > 0
        else 0
    )

    return {
        "documents": documents,
        "chunks": chunks,
        "embeddings": embeddings,
        "embedding_dimension": embedding_dimension
    }