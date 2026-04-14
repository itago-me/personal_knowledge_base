"""
RAG 问答管道:检索 + 生成2
"""
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import List, Dict, Any
from embedding import EmbeddingModel
from vector_store import VectorStore

class RAGPipeline:
    def __init__(self, 
                 persist_dir: str = "./data/chroma_db",
                 max_history = 5,
                 embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 llm_model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        初始化检索器和生成模型
        """
        # 初始化 embedding 模型
        self.embedder = EmbeddingModel(embedding_model_name)
        # 初始化向量数据库
        self.vector_store = VectorStore(persist_directory=persist_dir)
            # 增加基础存储会话记忆功能(保存ansewer,prompt)
        self.history = []
        self.max_history = max_history
        
        # 初始化本地大模型（用于生成）
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float32,   # 使用 float32 避免精度问题
            device_map="cpu",            # 强制 CPU，可改为 "auto" 自动选择 GPU
            trust_remote_code=True
        )
        self.model.eval()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文本块"""
        return self.vector_store.search(query, self.embedder, top_k)

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """基于检索结果生成答案"""
        # 构建上下文
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        # 构建提示词（Qwen2 的 chat 格式)
        messages = [
            {"role": "system", "content": "你是一个温柔、可爱、有耐心的AI助手。说话轻声细语,回答简洁友好"}]
        
        # 存入最进一次的会话
        for user_msg, assistant_msg in self.history[-self.max_history:]:
            messages.append({"role":"user","content":user_msg})
            messages.append({"role":"assistant","content":assistant_msg}) 

        # 添加上写文问题和上下文
        messages.append({"role": "user", "content": f"上下文:\n{context}\n\n问题:{query}\n\n请回答:"})

        # 应用 chat 模板
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 编码
        inputs = self.tokenizer(text, return_tensors ="pt", truncation=True, max_length=2048)
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # 解码
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        answer = generated_text

        # 更新历史
        self.history.append((query, answer))
        # 可选,限制历史长度
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
        return generated_text.strip()

    def ask(self, query: str, top_k: int = 5) -> str:
        """完整的问答流程"""
        # 1. 检索
        retrieved = self.retrieve(query, top_k)
        if not retrieved:
            return "未找到相关信息。"
        # 2. 生成
        answer = self.generate_answer(query, retrieved)
        return answer


# 测试
if __name__ == "__main__":
    rag = RAGPipeline()
    # question = "你是谁？"
    print("=" * 50)
    print(" 本地知识库聊天机器人已启动！")
    print(" 输入 'exit' 或 'quit' 退出聊天")
    print("=" * 50)

    while True:
        # 1. 获取用户输入（持续提问）
        question = input("\n请输入你的问题:")

        # 2. 退出条件
        if question.lower() in ["exit", "quit", "退出", "结束"]:
            print("聊天结束！")
            break

        # 3. 空输入跳过
        if not question.strip():
            continue
        answer = rag.ask(question)
        print(f"问题:{question}\n\n答案:{answer}")