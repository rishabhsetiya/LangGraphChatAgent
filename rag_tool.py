from langchain_core.tools import tool
import os
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

QDRANT_STORAGE_PATH = "C:\\Users\\RishabhSetiya\\qdrant_local_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "github_code_repo"
# Initialize Embedding Model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# Initialize Qdrant Client in local mode with persistent storage
qdrant_client = QdrantClient(path=QDRANT_STORAGE_PATH)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
def load_qdrant():
    return QdrantClient(path=QDRANT_STORAGE_PATH)  # embedded mode

@tool
def search_code(x: str) -> str:
    """Whenever any question related to code is asked this tool must be used"""
    result_str = ""
    # 1. Generate the query vector
    print(f"Embedding query: '{x}'")
    query_vector = embeddings.embed_query(x)

    # 2. Search the Qdrant Collection using the core 'query_points' method
    search_result = qdrant_client.query_points(  # <-- USE 'query_points'
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
        with_payload=True,
    )

    # 3. Process and Display Results
    result_str = result_str +"\n--- Top Search Results (Code Snippets) ---"
    for i, result in enumerate(search_result.points):  # Access the list of points via .points

        # Retrieve the metadata (payload)
        source_file = result.payload.get('source', 'N/A')
        repo_name = result.payload.get('repo_name', 'N/A')
        code_content = result.payload.get('text', 'No content')

        result_str = result_str + f"\nResult #{i + 1} (Score: {result.score:.4f})"
        result_str = result_str + f"  Repo: **{repo_name}** | File: **{source_file.split(os.path.sep)[-1]}**"
        result_str = result_str + "  Code Snippet:"

        snippet_display = code_content.split('\n')
        for line in snippet_display[:50]:
            result_str = result_str + f"    {line}"

    return result_str