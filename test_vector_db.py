from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

everconnect_vector_db_path = "vector_db"

hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(collection_name="everconnect_docs", embedding_function=hf_embed, persist_directory=everconnect_vector_db_path)

def get_similar_docs(question, similar_doc_count):
  return db.similarity_search(question, k=similar_doc_count)

# Let's test it with blackberries:
for doc in get_similar_docs("What is MAX day?", 3):
  print(doc.metadata['title'])
