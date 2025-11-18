# 删除API统一接口说明

## 概述

为了简化代码维护和提高接口一致性，我们创建了统一的删除API接口，支持删除文档和问答对，同时兼容单个删除和批量删除。两个接口都使用相同的后端处理逻辑，实现了代码复用。

## API接口

### 1. 单个删除

**接口地址**: `POST /kb/api/delete`

**请求体**:
```json
{
    "knowledge_base_name": "test",
    "document_name": "test_document",
    "delete_type": "document"  // "document" 或 "qa_pair"
}
```

**响应示例**:
```json
{
    "success": true,
    "message": "文档 'test_document' 已从知识库 'test' 中删除",
    "data": {
        "name": "test",
        "total_requested": 1,
        "success_count": 1,
        "failed_count": 0,
        "failed_items": null,
        "deleted_items": [
            {
                "document_name": "test_document.md",
                "type": "document",
                "operation_info": "删除操作详情"
            }
        ],
        "document_count": 4,
        "points_count": 20,
        "documents": ["other_doc.md", "qa_pairs.md"]
    }
}
```

### 2. 批量删除

**接口地址**: `POST /kb/api/delete/batch`

**请求体**:
```json
[
    {
        "knowledge_base_name": "test",
        "document_name": "document1",
        "delete_type": "document"
    },
    {
        "knowledge_base_name": "test",
        "document_name": "qa1",
        "delete_type": "qa_pair"
    },
    {
        "knowledge_base_name": "test",
        "document_name": "document2",
        "delete_type": "document"
    }
]
```

**响应示例**:
```json
{
    "success": true,
    "message": "批量删除完成：成功 3 个，失败 0 个",
    "data": {
        "name": "test",
        "total_requested": 3,
        "success_count": 3,
        "failed_count": 0,
        "failed_items": null,
        "deleted_items": [
            {
                "document_name": "document1.md",
                "type": "document",
                "operation_info": "删除操作详情"
            },
            {
                "document_name": "qa1.md",
                "type": "qa_pair",
                "operation_info": "删除操作详情"
            },
            {
                "document_name": "document2.md",
                "type": "document",
                "operation_info": "删除操作详情"
            }
        ],
        "document_count": 1,
        "points_count": 5,
        "documents": ["remaining_doc.md"]
    }
}
```

## 请求参数说明

### DeleteRequest 模型

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `knowledge_base_name` | string | ✅ | 知识库名称 |
| `document_name` | string | ✅ | 要删除的文档名或问答对名 |
| `delete_type` | string | ❌ | 删除类型，默认为 "document" |

### delete_type 可选值

- **`"document"`**: 删除文档（默认值）
- **`"qa_pair"`**: 删除问答对

## 技术实现

### 代码复用策略

1. **统一处理逻辑**: 两个API都调用 `delete_items_batch` 函数
2. **单个删除**: 将单个请求包装成列表，调用批量接口
3. **智能消息**: 根据删除数量和类型自动调整返回消息内容
4. **类型识别**: 根据 `delete_type` 字段区分文档和问答对

### 核心代码结构

```python
# 单个删除API（推荐使用）
@kb_router.post("/api/delete")
async def delete_item_unified(request: DeleteRequest):
    # 将单个请求包装成列表，调用批量接口
    batch_request = [request]
    return await delete_items_batch(batch_request)

# 批量删除API（兼容单个）
@kb_router.post("/api/delete/batch")
async def delete_items_batch(request: List[DeleteRequest]):
    # 统一的处理逻辑
    for delete_request in request:
        if delete_request.delete_type == "qa_pair":
            # 删除问答对
            operation_info = delete_by_source(document_name, vector_store)
        else:
            # 删除文档
            operation_info = delete_by_source(document_name, vector_store)
    
    # 根据删除数量调整返回消息
    if len(request) == 1:
        if request[0].delete_type == "qa_pair":
            message = f"问答对 '{request[0].document_name}' 已从知识库 '{knowledge_base_name}' 中删除"
        else:
            message = f"文档 '{request[0].document_name}' 已从知识库 '{knowledge_base_name}' 中删除"
    else:
        message = f"批量删除完成：成功 {success_count} 个，失败 {failed_count} 个"
```

## 使用场景

### 1. 删除单个文档
```python
# 删除名为 "report.pdf" 的文档
delete_request = {
    "knowledge_base_name": "medical_kb",
    "document_name": "report",
    "delete_type": "document"
}
```

### 2. 删除单个问答对
```python
# 删除名为 "faq.md" 的问答对
delete_request = {
    "knowledge_base_name": "medical_kb",
    "document_name": "faq",
    "delete_type": "qa_pair"
}
```

### 3. 批量删除混合类型
```python
# 同时删除文档和问答对
batch_delete = [
    {"knowledge_base_name": "medical_kb", "document_name": "old_report", "delete_type": "document"},
    {"knowledge_base_name": "medical_kb", "document_name": "outdated_qa", "delete_type": "qa_pair"},
    {"knowledge_base_name": "medical_kb", "document_name": "expired_doc", "delete_type": "document"}
]
```

### 4. 通过批量接口删除单个
```python
# 使用批量接口删除单个项目（完全兼容）
single_via_batch = [
    {"knowledge_base_name": "medical_kb", "document_name": "single_item", "delete_type": "document"}
]
```

## 优势

### 1. 代码维护
- **消除重复**: 不再需要维护两套相似的处理逻辑
- **统一更新**: 修改删除逻辑只需要在一个地方进行
- **减少bug**: 降低了因代码不一致导致的错误风险

### 2. 接口一致性
- **相同响应格式**: 两个接口返回相同的数据结构
- **统一错误处理**: 错误处理逻辑完全一致
- **一致的元数据**: 支持相同的元数据字段

### 3. 功能增强
- **类型识别**: 自动识别文档和问答对
- **混合删除**: 支持在同一批次中删除不同类型的项目
- **详细反馈**: 提供删除操作的详细信息

### 4. 性能优化
- **共享资源**: 两个接口共享GPU资源管理
- **批量处理**: 即使是单个删除，也能享受批量处理的优化

## 注意事项

### 1. 删除类型
- **文档删除**: `delete_type = "document"` 或省略（默认值）
- **问答对删除**: `delete_type = "qa_pair"`
- **自动后缀**: 系统会自动添加 `.md` 后缀

### 2. 错误处理
- **部分失败**: 批量删除中单个失败不会影响其他成功项
- **详细记录**: 记录每个失败项的具体错误信息
- **状态反馈**: 返回更新后的知识库状态

### 3. 资源管理
- **GPU优化**: 使用GPU资源管理器确保显存优化
- **批量处理**: 一次性处理多个删除请求，提高效率

## 测试

运行测试脚本验证接口功能：

```bash
python test_delete_unified.py
```

测试脚本会验证：
1. 单个文档删除
2. 单个问答对删除
3. 批量删除
4. 通过批量接口删除单个项目
5. 混合类型删除

## 总结

通过统一删除接口，我们实现了：
- ✅ 代码复用和维护简化
- ✅ 接口一致性和用户体验提升
- ✅ 功能增强和类型识别
- ✅ 性能优化和资源管理
- ✅ 向后兼容和平滑迁移

这种设计模式不仅适用于删除接口，也可以应用到其他类似的API接口中，提高整体代码质量和维护效率。
