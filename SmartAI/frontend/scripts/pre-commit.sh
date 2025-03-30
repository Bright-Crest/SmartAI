#!/bin/bash
# 无障碍检查 pre-commit 钩子
# 在代码提交前验证无障碍问题

# 获取当前暂存区中被修改的前端组件文件
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.tsx$|\.jsx$')

# 如果没有待提交的组件文件，直接通过检查
if [ -z "$STAGED_FILES" ]; then
  echo "✅ 没有待提交的组件文件，跳过无障碍检查"
  exit 0
fi

# 确保a11y-reports目录存在
mkdir -p frontend/a11y-reports

# 标记是否有严重的无障碍问题
HAS_CRITICAL_ISSUES=false

echo "🔍 正在检查无障碍问题..."

# 对每个暂存的组件文件进行无障碍检查
for FILE in $STAGED_FILES; do
  if [[ $FILE == frontend/components/* ]]; then
    echo "检查 $FILE"
    
    # 运行无障碍审计脚本
    yarn a11y:check:component "$FILE" > /dev/null
    
    # 检查最新报告中是否有严重问题
    if [ -f "frontend/a11y-reports/latest.json" ]; then
      CRITICAL_COUNT=$(cat frontend/a11y-reports/latest.json | grep -c '"severity": "critical"')
      
      if [ $CRITICAL_COUNT -gt 0 ]; then
        echo "⚠️ $FILE 中发现 $CRITICAL_COUNT 个严重无障碍问题"
        HAS_CRITICAL_ISSUES=true
      else
        echo "✅ $FILE 无严重无障碍问题"
      fi
    fi
  fi
done

# 如果存在严重问题，阻止提交
if [ "$HAS_CRITICAL_ISSUES" = true ]; then
  echo "❌ 提交被阻止：存在严重无障碍问题"
  echo "请修复这些问题，或使用 git commit --no-verify 强制提交"
  exit 1
else
  echo "✅ 无障碍检查通过"
  exit 0
fi 