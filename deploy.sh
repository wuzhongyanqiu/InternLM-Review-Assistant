#!/bin/bash
conda run -n interview-assistant

mkdir -p ./log

# ASR
nohup uvicorn server.asr.asr_server:app --host 0.0.0.0 --port 8001 > ./log/asr.out 2>&1 &

# TTS
nohup uvicorn server.tts.tts_server:app --host 0.0.0.0 --port 8002 > ./log/tts.out 2>&1 &

# InternVL
nohup lmdeploy serve api_server ./models/InternVL-Interview-Assistant --cache-max-entry-count 0.1 --backend turbomind --server-port 8003 --chat-template ./server/internvl/chat_template.json > ./log/internvl.out 2>&1 &

# InternLM
nohup lmdeploy serve api_server ./models/InternLM-Interview-Assistant --cache-max-entry-count 0.1 --model-name internlm2_5-7b-chat > ./log/internlm.out 2>&1 &
# lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat --cache-max-entry-count 0.1 --model-name internlm2_5-7b-chat

# RAG
nohup uvicorn server.tools.tools_server:app --host 0.0.0.0 --port 8004 > ./log/rag.out 2>&1 &
# cd lagent/server
# uvicorn rag_server:app --host 0.0.0.0 --port 8004 

echo "等待 ASR 响应"

while true; do
    if curl --silent -X GET http://localhost:8001/check; then
        echo "ASR 已响应"
        break
    else
        echo "ASR 尚未响应，等待 5 秒..."
        sleep 5
    fi
done

echo "等待 TTS 响应"

while true; do
    if curl --silent -X GET http://localhost:8002/check; then
        echo "TTS 已响应"
        break
    else
        echo "TTS 尚未响应，等待 5 秒..."
        sleep 5
    fi
done

echo "等待 InternLM-Interview-Assistant 响应"

while true; do
    if curl --silent -X GET http://localhost:8003/v1/models; then
        echo "InternLM-Interview-Assistant 已响应"
        break
    else
        echo "InternLM-Interview-Assistant 尚未响应，等待 5 秒..."
        sleep 5
    fi
done

echo "等待 tools 响应"

while true; do
  if curl --silent -X GET http://0.0.0.0:8004/check; then
    echo "tools 已响应"
    break
  else
    echo "tools 尚未响应，等待 5 秒..."
    sleep 5
  fi
done

echo "等待 InternVL-Interview-Assistant 响应"

# 首先获取模型名称
MODEL_NAME=$(curl -s -X POST -H "Authorization: YOUR_API_KEY" http://0.0.0.0:8005/v1/models/list | jq -r '.data[0].id')

# 然后使用模型名称发送 chat.completions 请求
while true; do
  if curl --silent -X GET "http://0.0.0.0:8005/v1/models"; then
    echo "InternVL-Interview-Assistant 已响应"
    break
  else
    echo "InternVL-Interview-Assistant 尚未响应，等待 5 秒..."
    sleep 5
  fi
done

echo "所有后端服务已启动并响应。"

# ps aux | grep lmdeploy
