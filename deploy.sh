#!/bin/bash
conda run -n interview-assistant

mkdir -p ./log

# InternLM
nohup lmdeploy serve api_server ./models/InternLM-Review-Assistant --cache-max-entry-count 0.1 --model-name internlm2_5-7b-chat > ./log/internlm.out 2>&1 &

cd lagent/server

# ASR
nohup uvicorn server.asr.asr_server:app --host 0.0.0.0 --port 8001 > ./log/asr.out 2>&1 &

# TTS
nohup uvicorn server.tts.tts_server:app --host 0.0.0.0 --port 8002 > ./log/tts.out 2>&1 &

# Digital_Human
nohup uvicorn digital_human_server:app --host 0.0.0.0 --port 8003 > ./log/ditial_human.out 2>&1 &

# RAG
nohup uvicorn server.tools.tools_server:app --host 0.0.0.0 --port 8004 > ./log/rag.out 2>&1 &

echo "等待 InternLM-Review-Assistant 响应"

while true; do
    if curl --silent -X GET http://localhost:23333/v1/models; then
        echo "InternLM-Interview-Assistant 已响应"
        break
    else
        echo "InternLM-Interview-Assistant 尚未响应，等待 5 秒..."
        sleep 5
    fi
done

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

echo "等待 Digital_Human 响应"

while true; do
  if curl --silent -X GET http://0.0.0.0:8003/check; then
    echo "Digital_Human 已响应"
    break
  else
    echo "Digital_Human 尚未响应，等待 5 秒..."
    sleep 5
  fi
done

echo "等待 RAG 响应"

while true; do
  if curl --silent -X GET http://0.0.0.0:8004/check; then
    echo "RAG 已响应"
    break
  else
    echo "RAG 尚未响应，等待 5 秒..."
    sleep 5
  fi
done

echo "所有后端服务已启动并响应。"

# ps aux | grep lmdeploy
