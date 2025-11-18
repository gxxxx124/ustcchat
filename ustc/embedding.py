from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.embeddings.base import Embeddings


class QwenEmbedding(Embeddings):
    def __init__(self, model_path: str = "/home/user/ustcchat/ustc/models/qwen_emb"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

    def _mean_pooling(self, outputs, attention_mask):
        token_embeddings = outputs.last_hidden_state  # 获取实际的 Tensor
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1),
                                                                                      min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,  # 根据模型调整最大长度
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """单个查询文本向量化"""
        return self.embed_documents([text])[0]