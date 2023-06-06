import pandas as pd
import shutil
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# Load and prepare knowledge base data
df = pd.read_csv("source_data/foundeverafter.csv")
df = df[['Link', 'Link-href', 'Description', 'TextField']]
df = df.fillna('none')
df.dropna(inplace=True)
df["combined"] = (
  "Title: " + df.Link.str.strip() + "; Description: " + df.Description.str.strip()+ "; Content: " + df.TextField.str.strip()
)
df = df[df["combined"].str.len() > 50]
df = df[df["Link"].str.len() > 0]
df.reset_index(level=0, inplace=True)

# Download model from Hugging face
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# DB set up
everconnect_vector_db_path = "vector_db"
shutil.rmtree(everconnect_vector_db_path)
documents = [Document(page_content=r["combined"], metadata={"source": r["Link-href"], "title": r["Link"]}) for _, r in df.iterrows()]

# Init the chroma db with the sentence-transformers/all-mpnet-base-v2 model loaded from hugging face  (hf_embed)
db = Chroma.from_documents(collection_name="everconnect_docs", documents=documents, embedding=hf_embed, persist_directory=everconnect_vector_db_path)
db.similarity_search("dummy") # tickle it to persist metadata (?)
db.persist()
