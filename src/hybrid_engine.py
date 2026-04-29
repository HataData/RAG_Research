import os
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer

class HybridSystem:
    def __init__(self):
        # 獲取當前檔案所在目錄的上一層 (根目錄)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(base_dir, "data", "paper.db")
        self.index_path = os.path.join(base_dir, "data", "faiss.bin")
        
        # 初始化
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.index = faiss.read_index(self.index_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ 引擎初始化成功，資料庫位置: {self.db_path}")

    def hybrid_search(self, query: str, k=3):
        # 向量檢索
        query_vec = self.model.encode([query]).astype('float32')
        _, indices = self.index.search(query_vec, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                # 確保 rowid 對應正確 (SQLite rowid 從 1 開始)
                cursor = self.conn.cursor()
                cursor.execute("SELECT content FROM docs WHERE rowid = ?", (int(idx) + 1,))
                row = cursor.fetchone()
                if row:
                    results.append(row[0])
        return results