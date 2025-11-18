# GitHub 上传详细步骤

## 📝 完整流程

### 第一步：在 GitHub 上创建新仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: 例如 `ustcchat`
   - **Description**: 例如 "智能对话系统 - USTC Chat"
   - **Visibility**: 选择 Public（公开）或 Private（私有）
   - ⚠️ **重要**：**不要**勾选以下选项：
     - ❌ Add a README file（我们已经有了）
     - ❌ Add .gitignore（我们已经有了）
     - ❌ Choose a license（可选，稍后添加）
3. 点击 **"Create repository"** 按钮

### 第二步：选择推送方式

创建仓库后，GitHub 会显示推送代码的说明。你可以选择：

#### 方式 A：HTTPS（简单，但每次需要输入密码）

```bash
cd /home/user/ustcchat

# 如果还没有初始化 Git
git init
git add .
git commit -m "Initial commit: USTC Chat project"

# 添加远程仓库（使用 GitHub 提供的 HTTPS URL）
git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

**注意**：推送时会提示输入：
- Username: 你的 GitHub 用户名
- Password: 需要使用 **Personal Access Token**（不是 GitHub 密码）

#### 方式 B：SSH（推荐，配置一次后无需输入密码）

**首先配置 SSH 密钥**：

```bash
# 1. 检查是否已有 SSH 密钥
ls -al ~/.ssh

# 2. 如果没有，生成新的 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"
# 按 Enter 使用默认路径，可以设置密码或直接按 Enter 跳过

# 3. 启动 SSH agent
eval "$(ssh-agent -s)"

# 4. 添加 SSH 密钥到 agent
ssh-add ~/.ssh/id_ed25519

# 5. 复制公钥内容
cat ~/.ssh/id_ed25519.pub
```

**然后在 GitHub 添加 SSH 密钥**：

1. 访问 https://github.com/settings/keys
2. 点击 **"New SSH key"**
3. **Title**: 填写一个名称，如 "My Server"
4. **Key**: 粘贴刚才复制的公钥内容（`cat ~/.ssh/id_ed25519.pub` 的输出）
5. 点击 **"Add SSH key"**

**最后推送代码**：

```bash
cd /home/user/ustcchat

# 如果还没有初始化 Git
git init
git add .
git commit -m "Initial commit: USTC Chat project"

# 添加远程仓库（使用 GitHub 提供的 SSH URL，格式：git@github.com:用户名/仓库名.git）
git remote add origin git@github.com:你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

## 🎯 推荐流程

**如果你是第一次使用**，建议：

1. ✅ 先在 GitHub 上创建仓库
2. ✅ 使用 **HTTPS 方式**先上传一次（简单直接）
3. ✅ 之后可以配置 SSH，方便后续操作

**如果你经常需要推送代码**，建议：

1. ✅ 先配置 SSH 密钥
2. ✅ 在 GitHub 上创建仓库
3. ✅ 使用 SSH 方式推送

## ❓ 常见问题

**Q: 如何获取 Personal Access Token？**
A: 
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" -> "Generate new token (classic)"
3. 设置权限（至少需要 `repo` 权限）
4. 生成后复制 Token（只显示一次）

**Q: 如何测试 SSH 连接？**
A: 
```bash
ssh -T git@github.com
# 如果看到 "Hi 用户名! You've successfully authenticated..." 说明配置成功
```

