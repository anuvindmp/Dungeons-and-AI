import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import pipeline

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARY_MODEL = "facebook/bart-large-cnn"

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
summarizer = pipeline("summarization", model=SUMMARY_MODEL)

def get_vectorstore(genre):
    db_path = f"./rag_store/{genre}"
    os.makedirs(db_path, exist_ok=True)
    return Chroma(persist_directory=db_path, embedding_function=embedding_model)

def seed_genre_documents(genre, folder_path):
    vector_db = get_vectorstore(genre)
    documents = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"genre": genre, "source": fname}))
    vector_db.add_documents(documents)
    vector_db.persist()

def retrieve_context(genre, query, k=3):
    vector_db = get_vectorstore(genre)
    docs = vector_db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def store_narrative(genre, narrative):
    vector_db = get_vectorstore(genre)
    summary = summarizer(narrative, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    doc = Document(page_content=narrative, metadata={"summary": summary, "type": "generated", "genre": genre})
    vector_db.add_documents([doc])
    vector_db.persist()
