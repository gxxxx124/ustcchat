# 问答对上传API统一接口说明

## 概述

为了简化代码维护和提高接口一致性，我们将原来的单个问答对上传和批量问答对上传API进行了统一。现在两个接口都使用相同的后端处理逻辑，实现了代码复用。

## API接口

### 1. 单个问答对上传

**接口地址**: `POST /kb/api/qa-pair`

**请求体**:
```json
{
    "knowledge_base_name": "test",
    "question": "什么是NGS检测？",
    "answer": "NGS（Next Generation Sequencing）是新一代测序技术...",
    "document_name": "test_qa.md",
    "metadata": {
        "category": "基因检测",
        "difficulty": "基础"
    }
}
```

**响应示例**:
```json
{
    "success": true,
    "message": "问答对已成功添加到知识库 'test'",
    "data": {
        "name": "test",
        "total_requested": 1,
        "success_count": 1,
        "failed_count": 0,
        "failed_items": null,
        "document_count": 5,
        "points_count": 25,
        "documents": ["test_qa.md", "other_doc.md"]
    }
}
```

### 2. 批量问答对上传

**接口地址**: `POST /kb/api/qa-pairs/batch`

**请求体**:
```json
[
    {
        "knowledge_base_name": "test",
        "question": "NGS检测的适用人群有哪些？",
        "answer": "NGS检测适用于：1. 遗传病患者及家属...",
        "document_name": "test_qa.md",
        "metadata": {
            "category": "基因检测",
            "difficulty": "中级"
        }
    },
    {
        "knowledge_base_name": "test",
        "question": "NGS检测的优势是什么？",
        "answer": "NGS检测的优势：1. 高通量...",
        "document_name": "test_qa.md",
        "metadata": {
            "category": "基因检测",
            "difficulty": "中级"
        }
    }
]
```

**响应示例**:
```json
{
    "success": true,
    "message": "批量上传完成：成功 2 个，失败 0 个",
    "data": {
        "name": "test",
        "total_requested": 2,
        "success_count": 2,
        "failed_count": 0,
        "failed_items": null,
        "document_count": 7,
        "points_count": 35,
        "documents": ["test_qa.md", "other_doc.md"]
    }
}
```

## 技术实现

### 代码复用策略

1. **统一处理逻辑**: 两个API都调用 `upload_qa_pairs_batch` 函数
2. **单个上传**: 将单个请求包装成列表，调用批量接口
3. **智能消息**: 根据请求数量自动调整返回消息内容

### 核心代码结构

```python
# 单个上传API（推荐使用）
@kb_router.post("/api/qa-pair")
async def upload_qa_pair_unified(request: QAPairRequest):
    # 将单个请求包装成列表，调用批量接口
    batch_request = [request]
    return await upload_qa_pairs_batch(batch_request)

# 批量上传API（兼容单个）
@kb_router.post("/api/qa-pairs/batch")
async def upload_qa_pairs_batch(request: List[QAPairRequest]):
    # 统一的处理逻辑
    # 根据上传数量调整返回消息
    if len(request) == 1:
        message = f"问答对已成功添加到知识库 '{knowledge_base_name}'"
    else:
        message = f"批量上传完成：成功 {success_count} 个，失败 {failed_count} 个"
```

## 优势

### 1. 代码维护
- **消除重复**: 不再需要维护两套相似的处理逻辑
- **统一更新**: 修改处理逻辑只需要在一个地方进行
- **减少bug**: 降低了因代码不一致导致的错误风险

### 2. 接口一致性
- **相同响应格式**: 两个接口返回相同的数据结构
- **统一错误处理**: 错误处理逻辑完全一致
- **一致的元数据**: 支持相同的元数据字段

### 3. 性能优化
- **共享资源**: 两个接口共享GPU资源管理
- **批量处理**: 即使是单个上传，也能享受批量处理的优化

## 使用建议

### 1. 新项目
- **推荐使用**: `/kb/api/qa-pair` 接口
- **简单直接**: 单个问答对上传的最佳选择

### 2. 批量场景
- **大量数据**: 使用 `/kb/api/qa-pairs/batch` 接口
- **批量导入**: 适合从文件或数据库批量导入问答对

### 3. 兼容性
- **向后兼容**: 原有的单个上传接口仍然可用
- **平滑迁移**: 可以逐步迁移到新的统一接口

## 测试

运行测试脚本验证接口功能：

```bash
python test_qa_unified.py
```

测试脚本会验证：
1. 单个问答对上传
2. 批量问答对上传
3. 通过批量接口上传单个问答对

## 注意事项

1. **知识库存在性**: 上传前会检查知识库是否存在
2. **错误处理**: 批量上传中单个失败不会影响其他成功项
3. **资源管理**: 使用GPU资源管理器确保显存优化
4. **元数据支持**: 支持自定义元数据字段

## 总结

通过统一问答对上传接口，我们实现了：
- ✅ 代码复用和维护简化
- ✅ 接口一致性和用户体验提升
- ✅ 性能优化和资源管理
- ✅ 向后兼容和平滑迁移

这种设计模式可以应用到其他类似的API接口中，提高整体代码质量和维护效率。
