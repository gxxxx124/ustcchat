# 智能搜索引擎 - 按需健康检查机制

## 概述

本系统已从定期健康检查改为按需健康检查，以**避免不必要的token消耗**，特别是Tavily API的搜索token。

## 主要改进

### 1. 移除定期健康检查
- ❌ 原来：每30秒自动执行健康检查
- ✅ 现在：只在真正需要时才执行健康检查

### 2. 按需触发条件
健康检查会在以下情况下自动触发：

- **首次使用**：系统初始化后的第一次搜索
- **后端全部不可用**：所有搜索后端都标记为不健康
- **错误次数过多**：某个后端连续失败3次以上
- **超时检查**：距离上次健康检查超过5分钟（最大间隔）

### 3. 智能触发逻辑
```python
def _should_check_health(self) -> bool:
    # 如果从未检查过，需要检查
    if self.last_health_check == 0:
        return True
    
    # 如果距离上次检查超过最大间隔，需要检查
    if current_time - self.last_health_check > self.health_check_interval:
        return True
    
    # 如果所有后端都标记为不健康，需要检查
    if not any(status["healthy"] for status in self.backend_status.values()):
        return True
    
    # 如果某个后端错误次数过多，需要检查
    for backend, status in self.backend_status.items():
        if status["error_count"] >= 3:  # 连续3次错误后强制检查
            return True
    
    return False
```

## 配置参数

```python
# 健康检查配置 - 按需执行，不定期执行
self.health_check_interval = 300  # 5分钟，仅作为最大间隔
```

## 使用方法

### 自动触发
健康检查会在需要时自动执行，无需手动干预。

### 手动触发
如果需要手动执行健康检查：

```python
from smart_search import smart_search_engine

# 手动触发健康检查
result = smart_search_engine.force_health_check()
print(f"健康检查结果: {result}")
```

### 状态监控
```python
# 获取搜索引擎状态
status = smart_search_engine.get_status()
print(f"后端状态: {status['backend_status']}")
print(f"上次健康检查: {status['last_health_check']}")
```

## 测试

运行测试脚本验证按需健康检查机制：

```bash
python test_smart_search_health.py
```

## 优势

1. **节省Token**：避免每30秒消耗Tavily搜索token
2. **智能检测**：只在真正需要时才检查后端健康状态
3. **快速响应**：后端故障时能快速检测并恢复
4. **资源优化**：减少不必要的API调用和网络请求

## 注意事项

1. **首次使用**：系统启动后的第一次搜索会执行健康检查
2. **故障恢复**：后端故障后，系统会自动尝试恢复并重新检查
3. **手动检查**：如需要，可以调用`force_health_check()`手动执行
4. **监控日志**：健康检查的执行会在日志中记录，便于监控

## 故障转移机制

当某个后端失败时：

1. 标记后端为不健康
2. 增加错误计数
3. 尝试其他可用后端
4. 如果所有后端都不可用，触发健康检查
5. 尝试恢复搜索服务

这种机制确保了系统的高可用性，同时最小化了资源消耗。
