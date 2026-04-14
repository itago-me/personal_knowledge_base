# 文本切分模块：将长文档切分成适合向量检索的小块

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    # 使用递归字符分割器，智能切分文本
    
    def __init__(self,chunk_size: int = 500,chunk_overlap: int = 50):
        """
        :param chunk_size: 每个块的最大字符数
        :param chunk_overlap: 块之间的重叠字符数(保持上下文连续)
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function = len,
        )

    def split_document(self,document:List[Dict[str, Any]]) ->List[Dict[str, Any]]:
        """
        对文档列表进行分块
        :param documents: 每个元素包含'content'和'source'的字典
        :return 分块后的列表,每个元素增加'chunk_id'和'chunk_set'
        """
        chunks = []
        for doc in document:
            # 获取原始内容
            text = doc["content"]
            # 切分
            chunk_texts = self.splitter.split_text(text)
            # 构建分块结果
            for idx, chunk_text in enumerate(chunk_texts):
                chunks.append({
                    "source": doc["source"],
                    "chunk_id": idx,
                    "chunk_text": chunk_text,
                })
        return chunks
            
# 简单的测试(当直接运行此文件时)
if __name__ == "__main__":
    from document_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_all()
    if not docs:
        print("没有加载的任何文档,请检查date/raw目录")
    else:
        splitter = TextSplitter(chunk_size = 100, chunk_overlap = 20)
        chunks = splitter.split_document(docs)
        print(f"共生成{len(chunks)}个文本块")
        for i, chunk in enumerate(chunks[:3]):  #只显示3块
            print(f"\n块{i + 1}(来源:{chunk['source']})")
            print(f"内容:{chunk['chunk_text'][: 100]}...")