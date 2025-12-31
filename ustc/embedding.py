from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.embeddings.base import Embeddings
import os


class QwenEmbedding(Embeddings):
    def __init__(self, model_path: str = None):
        # 允许通过环境变量覆盖模型路径，默认指向本地已下载的Qwen embedding目录
        self.model_path = model_path or os.getenv(
            "EMBEDDING_MODEL_PATH",
            "/home/user/ustcchat/ustc/models/Qwen"  # 实际存在的模型目录
        )
        
        # 检查是否是sentence-transformers格式的模型
        if os.path.exists(os.path.join(self.model_path, "config_sentence_transformers.json")):
            # 使用sentence-transformers加载（Qwen3-Embedding-0.6B）
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_path)
                self.use_sentence_transformers = True
                print(f"✅ 使用sentence-transformers加载模型: {self.model_path}")
            except ImportError:
                print("⚠️ sentence-transformers未安装，回退到transformers方式")
                self.use_sentence_transformers = False
                self._load_with_transformers()
        else:
            # 使用transformers加载（旧的Qwen2.5模型）
            self.use_sentence_transformers = False
            self._load_with_transformers()
    
    def _load_with_transformers(self):
        """使用transformers加载模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
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
        if self.use_sentence_transformers:
            # 使用sentence-transformers
            embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.tolist()
        else:
            # 使用transformers + mean pooling
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
