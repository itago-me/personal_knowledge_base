"""
FastAPI 后端服务，提供文档上传、重建索引、问答等接口
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import shutil   #
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# 导入项目内部的模块
from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline  # 需要实现 ask 方法
from pydantic import BaseModel

# 路径配置
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"

# 确保必要目录存在
RAW_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# 初始化 FastAPI 应用
app = FastAPI(title="个人知识库问答系统", version="1.0")

# 可选：提供静态文件和模板（如果想用简单的HTML页面）
# 注意：如果你不打算使用 HTML 页面，可以不用这部分
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
# 全局 RAG 实例（懒加载，避免每次请求都初始化）
_rag_pipeline = None


# 向量检索加强生成的实例(调用时才会进行模型的下载)
def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            persist_dir=str(CHROMA_DIR),
            embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            llm_model_name="Qwen/Qwen2-0.5B-Instruct"
        )
    return _rag_pipeline

# ------------------- 辅助函数 -----------------俄--
def rebuild_index():
    """重建向量索引（基于当前 raw 目录下的所有文件）"""
    loader = DocumentLoader(data_dir=str(RAW_DIR))
    docs = loader.load_all()
    if not docs:
        raise HTTPException(status_code=400, detail="没有找到文档，请先上传文件")
    
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_document(docs)

    embedder = EmbeddingModel()
    texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings = embedder.embed(texts)
    store = VectorStore(persist_directory=str(CHROMA_DIR))
    # 注意：这里可能需要先清空旧数据，我们直接覆盖（使用相同 collection 会追加）
    # 简单起见，每次重建前可以删除 collection 重新创建，但这里我们暂时追加
    store.add_chunks(chunks, embeddings.tolist())
    return len(chunks)

# ------------------- API 端点 -------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回简单的 HTML 前端页面"""

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>个人知识库问答系统</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: auto; }
            textarea { width: 100%; height: 100px; }
            input, button { margin: 5px; }
            .result { margin-top: 20px; border: 1px solid #ccc; padding: 10px; background: #f9f9f9; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>个人知识库问答系统</h1>
            <hr>

            <h2>1. 上传文档</h2>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="files" multiple>
                <button type="submit">上传</button>

            </form>
            <hr>
            <h2>2. 重建索引</h2>
            <button onclick="rebuildIndex()">重建索引</button>
            <div id="rebuildMsg"></div>
            <hr>
            <h2>3. 提问</h2>
            <textarea id="query" placeholder="请输入你的问题..."></textarea>
            <button onclick="ask()">提问</button>
            <div id="answer" class="result"></div>
        </div>
        <script>
            async function rebuildIndex() {
                const msgDiv = document.getElementById('rebuildMsg');
                msgDiv.innerText = '正在重建索引...';
                const response = await fetch('/rebuild', { method: 'POST' });
                const data = await response.json();
                msgDiv.innerText = data.message;
            }
            async function ask() {
                const query = document.getElementById('query').value;
                if (!query) return;
                const answerDiv = document.getElementById('answer');
                answerDiv.innerText = '正在思考...';
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await response.json();
                answerDiv.innerText = data.answer;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上传一个或多个文档到 raw 目录"""
    saved_files = []
    for file in files:
        if file.filename:
            file_path = RAW_DIR / file.filename
            # 防止覆盖
            if file_path.exists():
                # 可以选择覆盖或跳过，这里覆盖
                pass
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
    return {"message": f"成功上传 {len(saved_files)} 个文件", "files": saved_files}

@app.post("/rebuild")
async def rebuild():
    """重建向量索引"""
    try:
        chunk_count = rebuild_index()
        return {"message": f"索引重建成功，共处理 {chunk_count} 个文本块"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 更通用的 JSON 请求方式
class AskRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(req: AskRequest):
    """提问接口（使用表单数据）"""
    # 如果使用 JSON 请求，可以改成接收 body 参数
    rag = get_rag_pipeline()
    answer = rag.ask(req.query)
    return {"answer": answer}
    # pass





@app.post("/ask_json")
async def ask_json(req: AskRequest):
    """提问接口(JSON 格式）"""
    rag = get_rag_pipeline()
    # rag = RAGPipeline()
    answer = rag.ask(req.query)
    return {"answer": answer}

# 同时保留表单版本
@app.post("/ask_form") 
async def ask_form(query: str = Form(...)):
    rag = get_rag_pipeline()
    answer = rag.ask(query)
    return {"answer": answer}