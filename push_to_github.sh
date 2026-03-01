#!/bin/bash

# GitHub仓库推送脚本
# 使用方法: bash push_to_github.sh YOUR_GITHUB_USERNAME

if [ -z "$1" ]; then
    echo "错误: 请提供GitHub用户名"
    echo "使用方法: bash push_to_github.sh YOUR_GITHUB_USERNAME"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME="flash-attention-modeling"

echo "═══════════════════════════════════════════════════════════"
echo "  推送到GitHub: $GITHUB_USERNAME/$REPO_NAME"
echo "═══════════════════════════════════════════════════════════"

# 检查是否已有remote
if git remote | grep -q origin; then
    echo "⚠️  检测到已存在的remote，将先移除..."
    git remote remove origin
fi

# 添加remote
echo "📝 添加remote..."
git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# 重命名分支为main
echo "🔄 重命名分支为main..."
git branch -M main

# 推送
echo "🚀 推送到GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 成功推送到GitHub!"
    echo "🔗 仓库地址: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
else
    echo ""
    echo "❌ 推送失败，请检查:"
    echo "   1. GitHub仓库是否已创建"
    echo "   2. 用户名是否正确"
    echo "   3. 是否有推送权限"
    echo ""
    echo "💡 如果仓库不存在，请先访问:"
    echo "   https://github.com/new"
    echo "   创建名为 '$REPO_NAME' 的仓库"
fi
