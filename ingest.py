from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import sqlite3
import faiss
import numpy as np

# 1. 載入 PDF
loader = PyPDFLoader("data/230608045v2.pdf")
docs = loader.load()

# 2. 分割文字 (這一步很重要)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 準備模型與資料庫
model = SentenceTransformer('all-MiniLM-L6-v2')
conn = sqlite3.connect("data/paper.db")
conn.execute("CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, content TEXT)")

# 4. 寫入資料庫並建立索引
embeddings = []
for i, chunk in enumerate(chunks):
    text = chunk.page_content
    conn.execute("INSERT INTO docs (content) VALUES (?)", (text,))
    embeddings.append(model.encode(text))

conn.commit()
conn.close()

# 5. 建立 FAISS 索引
index = faiss.IndexFlatL2(384) # 384 是 all-MiniLM 的維度
index.add(np.array(embeddings).astype('float32'))
faiss.write_index(index, "data/faiss.bin")

print("✅ 論文資料已成功入庫！")