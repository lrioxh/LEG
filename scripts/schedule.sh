#!/bin/bash

# 定义要执行的脚本路径
TARGET_SCRIPT="/home/lrioxh/code/LEG/scripts/run_log.sh"

# 定义运行时间
TIME="6:30"

# 使用 at 调度任务
at $TIME <<EOF
bash $TARGET_SCRIPT
EOF

echo "Scheduled $TARGET_SCRIPT to run at $TIME"