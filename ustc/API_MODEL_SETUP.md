# API模型配置说明

## 为什么使用API模型？

1. **更智能的推荐逻辑**：API模型（如GPT-4o）具有更强的语义理解能力，能更好地匹配"有机材料光电性能"与"RSoXS有机光电"的关联
2. **更好的工具调用**：能更准确地决定何时调用工具以及如何构造查询参数
3. **更强的推理能力**：在复杂的技术领域匹配和优先级判断上表现更佳

## 如何切换到API模型

### 方法1：环境变量配置

```bash
# 设置使用API模型
export MODEL_TYPE=api
export OPENAI_API_KEY=your_api_key_here
export OPENAI_API_BASE=https://api.openai.com/v1  # 可选，默认OpenAI
export OPENAI_MODEL=gpt-4o-mini  # 可选，默认gpt-4o-mini

# 启动服务
conda activate langchain
python web_memory.py
```

### 方法2：支持的API提供商

#### OpenAI
```bash
export MODEL_TYPE=api
export OPENAI_API_KEY=sk-...
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini
```

#### 其他兼容OpenAI API的提供商
```bash
export MODEL_TYPE=api
export OPENAI_API_KEY=your_key
export OPENAI_API_BASE=https://your-provider.com/v1
export OPENAI_MODEL=your-model-name
```

### 方法3：回退到本地模型

```bash
# 使用本地Ollama模型（默认）
export MODEL_TYPE=ollama
# 或者不设置环境变量
python web_memory.py
```

## 推荐配置

对于线站推荐系统，推荐使用以下配置：

```bash
export MODEL_TYPE=api
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o-mini  # 性价比高，推荐逻辑好
```

## 日志改进

现在日志会详细显示：
- 🔧 工具执行过程
- 📤 工具返回内容长度和预览
- 🎯 工具返回完整内容
- 📝 用户问题和工具结果的整合

查看日志：
```bash
tail -f logs/chat_logs_$(date +%Y%m%d).log
```

## 测试建议

1. 先测试本地模型：`export MODEL_TYPE=ollama`
2. 再测试API模型：`export MODEL_TYPE=api`
3. 对比两种模型的推荐结果和工具调用质量

