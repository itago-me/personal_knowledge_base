# src/main.py
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from embedding import EmbeddingModel
from vector_store import VectorStore

def main():
    # 1. 加载文档
    loader = DocumentLoader()
    docs = loader.load_all()
    if not docs:
        print("没有找到文档，请检查 data/raw 目录")
        return
    print(f"加载了 {len(docs)} 个文档")

    # 2. 分块
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_document(docs)
    print(f"共生成 {len(chunks)} 个文本块")

    # 3. 生成向量
    embedder = EmbeddingModel()
    texts = [chunk['chunk_text'] for chunk in chunks]
    embeddings = embedder.embed(texts)
    print(f"生成向量维度: {embeddings.shape}")

    # 4. 存入数据库
    store = VectorStore()
    store.add_chunks(chunks, embeddings.tolist())

    # 5. 测试检索
    query = "我是谁?"
    results = store.search(query, embedder, top_k=3)
    print("\n检索结果:")
    for res in results:
        print(f"- 距离: {res['distance']:.4f} | 来源: {res['metadata']['source']}")
        print(f"  文本: {res['text'][:100]}...\n")

if __name__ == "__main__":
    main()