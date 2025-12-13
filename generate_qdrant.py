import os

from dotenv import load_dotenv
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from qdrant_client import models, QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

REPO_URLS = [
    "https://github.com/rishabhsetiya/leadmanagement.git"
]

CLONE_DIR_BASE = "cloned_repos"

QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

all_documents = []

for repo_url in REPO_URLS:
    # 1. Define local clone directory for the specific repo
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    clone_path = os.path.join(CLONE_DIR_BASE, repo_name)

    print(f"--- Processing repository: {repo_name} ---")

    # 2. Clone the repository locally using LangChain's GitLoader
    # This uses GitPython internally to clone the repository
    try:
        # Note: If the directory already exists, it will try to load from it.
        # If you want to force a fresh clone, you may need to delete the
        # directory first or use the 'branch' parameter.
        loader = GitLoader(
            repo_path=clone_path,
            clone_url=repo_url,
            branch="master",  # Adjust branch name as necessary
        )
        print(f"Loading documents from {repo_url}...")
        repo_docs = loader.load()
        all_documents.extend(repo_docs)
        print(f"Loaded {len(repo_docs)} documents.")
    except Exception as e:
        print(f"Error processing {repo_url}: {e}")
        continue

if not all_documents:
    print("No documents were loaded. Exiting.")
    exit()

# 3. Chunk the documents for better retrieval
# A good chunking strategy is vital for code. This configuration is a decent starting point.
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA,
    chunk_size=2000,
    chunk_overlap=200
)

# Split all documents from all repos
chunks = text_splitter.split_documents(all_documents)
print(f"\nTotal chunks created: {len(chunks)}")

# 4. Use the Qdrant Client to Explicitly Create/Recreate the Collection
print(f"Manually recreating collection '{COLLECTION_NAME}'...")

# Initialize Qdrant Client in local mode with persistent storage
qdrant_client = QdrantClient(path=QDRANT_STORAGE_PATH)
# Use 'recreate_collection' which will delete the old one if it exists,
# then create a new one, avoiding the "not found" or lock errors.
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
print("Collection recreation successful.")

print("Generating vectors for all documents...")
vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
vector_size = len(vectors[0])

# 5. Prepare Points (Vectors + Metadata Payload)
points = []
for i, chunk in enumerate(chunks):
    points.append(
        models.PointStruct(
            id=i,
            vector=vectors[i],
            # Combine the chunk content and its metadata into the Qdrant payload
            payload={
                "text": chunk.page_content,
                **chunk.metadata
            }
        )
    )

# 6. Upsert (Upload) Points
print(f"Uploading {len(points)} points to Qdrant...")
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,
    points=points
)

qdrant_client.close()

print("\n✅ Qdrant Vector Store creation and population complete.")
print(f"Data is persisted locally at: **{os.path.abspath(QDRANT_STORAGE_PATH)}**")