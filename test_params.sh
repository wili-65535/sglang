#!/bin/bash

# 测试num_inference_steps参数是否生效

echo "创建视频生成任务（27步）..."
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat walks on the grass, realistic",
    "size": "720x1280",
    "seconds": 5,
    "fps": 16,
    "num_frames": 81,
    "seed": 1234,
    "num_inference_steps": 27,
    "guidance_scale": 3.5,
    "guidance_scale_2": 4.0
  }')

echo "响应: $RESPONSE"

# 提取video_id
VIDEO_ID=$(echo $RESPONSE | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
echo "任务ID: $VIDEO_ID"

if [ -z "$VIDEO_ID" ]; then
    echo "错误：未能创建任务"
    exit 1
fi

echo "等待生成完成，观察日志中的步数..."
echo "如果看到 '0%|  | 0/27' 而不是 '0%|  | 0/40'，说明参数生效了"
