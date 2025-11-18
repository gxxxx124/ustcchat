# GitHub 上传指南

## ✅ 已完成的修改

1. **敏感信息已改为从环境变量读取**：
   - `ustc/auth.py`: JWT 密钥从环境变量读取
   - `ustc/ddb.py`: OSS 配置从环境变量读取
   - `ustc/web_memory.py`: 数据库连接字符串从环境变量读取
   - `ustc/web_api.py`: 数据库连接字符串从环境变量读取

2. **已创建的文件**：
   - `.gitignore`: Git 忽略文件
   - `README.md`: 项目说明文档
   - `env.example`: 环境变量配置模板

## 📋 上传到 GitHub 的步骤

### 方法一：使用 Git 命令行（推荐）

1. **初始化 Git 仓库**（如果还没有）：
   ```bash
   cd /home/user/ustcchat
   git init
   ```

2. **添加所有文件**：
   ```bash
   git add .
   ```

3. **提交代码**：
   ```bash
   git commit -m "Initial commit: USTC Chat project"
   ```

4. **在 GitHub 上创建新仓库**：
   - 访问 https://github.com/new
   - 填写仓库名称（如 `ustcchat`）
   - 选择 Public 或 Private
   - **不要**初始化 README、.gitignore 或 license（我们已经有了）
   - 点击 "Create repository"

5. **添加远程仓库并推送**：
   ```bash
   git remote add origin https://github.com/你的用户名/仓库名.git
   git branch -M main
   git push -u origin main
   ```

   如果使用 SSH：
   ```bash
   git remote add origin git@github.com:你的用户名/仓库名.git
   git branch -M main
   git push -u origin main
   ```

### 方法二：使用 GitHub Desktop

1. 下载并安装 GitHub Desktop
2. 登录你的 GitHub 账号
3. 点击 "File" -> "Add Local Repository"
4. 选择 `/home/user/ustcchat` 目录
5. 点击 "Publish repository" 上传到 GitHub

## 🔐 关于认证

**你不需要提供账号密码给我**。上传到 GitHub 有两种方式：

1. **HTTPS 方式**：首次推送时会提示输入 GitHub 用户名和密码（或 Personal Access Token）
2. **SSH 方式**：需要配置 SSH 密钥，之后无需输入密码

### 配置 SSH 密钥（推荐）

```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 查看公钥
cat ~/.ssh/id_ed25519.pub

# 复制公钥内容，然后：
# 1. 访问 https://github.com/settings/keys
# 2. 点击 "New SSH key"
# 3. 粘贴公钥内容
# 4. 保存
```

## ⚠️ 上传前检查清单

- [x] 敏感信息已改为从环境变量读取
- [x] `.gitignore` 已创建
- [x] `README.md` 已创建
- [x] `env.example` 已创建
- [ ] 确认没有硬编码的真实密钥
- [ ] 确认日志文件不会被上传（已在 .gitignore 中）
- [ ] 确认数据库文件不会被上传（已在 .gitignore 中）
- [ ] 确认模型文件不会被上传（已在 .gitignore 中）

## 📝 环境变量配置

上传后，其他用户需要：

1. 复制 `env.example` 为 `.env`
2. 填写真实的环境变量值
3. 运行应用

## 🚀 后续步骤

上传成功后，建议：

1. 在 GitHub 仓库设置中添加描述
2. 添加 Topics（标签）如：`fastapi`, `langgraph`, `rag`, `chatbot`
3. 考虑添加 LICENSE 文件
4. 设置 GitHub Actions 进行 CI/CD（可选）

