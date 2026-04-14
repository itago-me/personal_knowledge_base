"""
向量化模块: 将文本转化为稠密变量
"""

from sentence_transformers import SentenceTransformer
import numpy as np

from typing import List, Union


class EmbeddingModel:
    def __init__(self, model_name:str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化embeddign模型
        :param moder_name: 模型名称(支持中文的轻量级模型)
        """

        self.model = SentenceTransformer(model_name)


    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        将文本转化为向量
        :param texts: 单个文本或文本列表
        :return 向量数组: shape = (len(texts), embeddign_dim)
        """

        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy = True)
        return embeddings
    
if __name__ == "__main__":
    model = EmbeddingModel()
    test_text = "这是一个测试句子"
    vec = model.embed(test_text)
    print(f"向量维度:{vec.shape}")
        