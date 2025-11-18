# root_path 配置说明

## 问题说明

当访问 `http://localhost:8000/nsrlchat/` 时，如果设置了 `root_path="/nsrlchat"`，会导致路径处理问题。

## 解决方案

`root_path` 现在通过环境变量 `ROOT_PATH` 配置：

### 场景1：本地直接测试（不使用nginx）

**不需要设置 `ROOT_PATH`**，直接访问：
- `http://localhost:8000/`
- `http://localhost:8000/auth/login-page`

### 场景2：通过nginx反向代理部署

**需要设置 `ROOT_PATH` 环境变量**：

```bash
export ROOT_PATH="/nsrlchat"
```

然后通过nginx配置：
```nginx
location /nsrlchat/ {
    proxy_pass http://localhost:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Prefix /nsrlchat;
}
```

访问：
- `https://nsrloa.ustc.edu.cn/nsrlchat/`
- `https://nsrloa.ustc.edu.cn/nsrlchat/auth/login-page`

## 当前状态

- ✅ FileResponse 路径已修复（使用绝对路径）
- ✅ root_path 可通过环境变量配置
- ✅ 默认不使用 root_path（方便本地测试）

## 测试建议

1. **本地测试**：直接访问 `http://localhost:8000/`（不要加 `/nsrlchat`）
2. **生产部署**：设置 `ROOT_PATH=/nsrlchat` 并通过nginx访问

