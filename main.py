from fastapi import FastAPI, HTTPException
from src.hybrid_engine import HybridSystem

app = FastAPI(title="RAG Paper Search API")

# 全域初始化，確保只載入一次模型
try:
    engine = HybridSystem()
except Exception as e:
    print(f"❌ 初始化失敗: {e}")
    engine = None

@app.get("/hybrid_search")
async def search(query: str, alpha: float = 0.5):
    if not engine:
        raise HTTPException(status_code=500, detail="搜尋引擎未初始化")
    
    try:
        results = engine.hybrid_search(query)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)