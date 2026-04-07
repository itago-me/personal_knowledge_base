"""
文档加载模块：支持 txt, pdf, markdown 等格式
"""
from pathlib import Path
from typing import List, Dict, Any
import pypdf
import markdown
from docx import Document as DocxDocument

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

class DocumentLoader:
    """统一文档加载器"""
    


    def __init__(self, data_dir: str = None):
        if data_dir is None:
            self.data_dir = DATA_DIR
        else:
            self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    
    def load_all(self) -> List[Dict[str, Any]]:
        """加载 data_dir 下所有支持的文档"""
        documents = []
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                content = self._load_file(file_path)
                if content:
                    documents.append({
                        "source": str(file_path),
                        "content": content
                    })
        return documents
    
    def _load_file(self, file_path: Path) -> str:
        """根据文件扩展名调用对应解析器"""
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".txt":
                return file_path.read_text(encoding='utf-8')
            elif suffix == ".pdf":
                return self._load_pdf(file_path)
            elif suffix in [".md", ".markdown"]:
                return self._load_markdown(file_path)
            elif suffix == ".docx":
                return self._load_docx(file_path)
            else:
                print(f"暂不支持的文件格式: {suffix}")
                return ""
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            return ""
    
    def _load_pdf(self, file_path: Path) -> str:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    
    def _load_markdown(self, file_path: Path) -> str:
        # 先读原始文本，后续可以保留格式或转成纯文本
        md_text = file_path.read_text(encoding='utf-8')
        # 如果需要纯文本，可以用 markdown 库转 html 再提取，但这里先保留原始文本
        return md_text
    
    def _load_docx(self, file_path: Path) -> str:
        doc = DocxDocument(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()


# 简单的测试（当直接运行此文件时）
if __name__ == "__main__":
#     print("yes")
    loader = DocumentLoader()
    docs = loader.load_all()
    for i, doc in enumerate(docs):
        print(f"文档 {i+1}: {doc['source']}")
        print(f"内容预览: {doc['content'][:100]}...\n")
