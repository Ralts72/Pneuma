#!/bin/bash

# 配置：PC WSL2 信息
PC_USER="ralts"
PC_IP="192.168.1.11"
PROJECT_DIR="~/project/Pneuma"

# 日志文件名（带时间戳）source ${PROJECT_DIR}/.venv/bin/activate &&
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

# SSH 执行命令
ssh ${PC_USER}@${PC_IP} "
  cd ${PROJECT_DIR} &&
  mkdir -p logs &&
  git pull origin main &&
  nohup python3 main.py > ${LOG_FILE} 2>&1 &
"

echo "训练任务已发送到 PC，日志保存在 ${PROJECT_DIR}/${LOG_FILE}"