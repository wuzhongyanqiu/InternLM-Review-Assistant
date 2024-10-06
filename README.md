# InternLM-Review-Assistant
基于 InternLM 的复习助手项目，欢迎大家也来参加书生大模型实战营项目[http://github.com/internLM/tutorial](http://github.com/internLM/tutorial)

## 简介
对于内向者而言，复习常成负担，难寻24小时陪伴的复习伙伴。基于 InternLM 的个人复习小助手应运而生，专为你解决这一难题。只需上传笔记和知识文件，小助手即刻解析，生成问题库与知识库，助你高效复习。选择快问快答模式，小助手将根据你的文件提问并评估答案；选择模拟面试，小助手则根据你的简历和知识文件，为你量身定制面试体验。借助 XTuner、LMDeploy、Lagent 框架，结合 RAG 与 Agent 技术，小助手将成为你复习路上的得力助手。

## 项目功能
用户上传个人文件，复习助手解析文件，建立知识库和面试题库，Agent 功能使得复习助手根据场景进行相应地行动，具体为简历查看、关键词抽题、随机抽题、答案评估，支持语音输入和输出。

## 架构图

![架构图](./assets/architecture_diagram.jpeg)

## 演示视频

## 快速开始
1. clone 本项目至本地开发机
```bash
git clone https://github.com/wuzhongyanqiu/InternLM-Review-Assistant.git
```

2. 配置环境
```
# 创建虚拟环境
conda create -n review-assistant python=3.10
# 激活环境
conda activate review-assistant
# 安装所需依赖
cd InternLM-Review-Assistant
pip install -r requirements.txt
```

3. 启动
```
python app.py
```

4. 示例效果

## 思路讲解视频

## 微调数据集构建流程

1. 处理知识文件
- 本项目使用的知识文件均来自开源社区中的教材、文档。
- [MinerU](https://github.com/opendatalab/MinerU) 将 PDF 文件转换为 markdown 格式。
- 切分文档，利用 InternLM 和 Faiss 生成问题数据库和知识文件向量库。

2. 关键词查找问题，RAG 答案评估
- 关键词查找问题使用 SQL 语句和 InternLM 筛选。
- 答案评估使用 BM25Retriever 和 FaissRetriever，采取多路召回模式，向量模型分别用的是 m3e、gte、bge、bce。

3. 生成虚拟简历
- 使用 GPT4O 生成符合格式要求的虚拟简历，在读取过程中再次利用 InternLM 摘要总结。

4. 构建多轮对话数据，合成数据来源于 InternLM & DeepSeek
- 构建多轮对话指令数据集，其具体数据格式分两种，一是快问快答多轮对话格式(3k)，二是模拟面试多轮对话格式(2k)
- 数据格式为：
```
[
    {
        "conversation": [
            {
                "system": "",
                "input": "",
                "output": ""
            },
            {
                "input": "",
                "output": ""
            },
            {
                "input": "",
                "output": ""
            },
            ...
        ]
    }
    ...
]
```
- 详细数据见 finetune/finetune_data.json

## 微调流程
1. 将`./finetune/internlm2_chat_7b_qlora_interview_data.py`中的数据集路径和模型路径替换为本地路径，其余参数根据需求和资源调整。
由于微调数据集格式特殊，将 XTuner/xtuner/engine/hooks/evaluate_chat_hook.py:
```
inputs = (self.system + self.instruction).format(
                input=sample_input, round=1, **runner.cfg)
```
改成了
```
inputs = self.system + self.instruction.format(
                input=sample_input, round=1, **runner.cfg)
```
2. 使用命令进行训练，自定义评估问题，可以手动早停：
```
xtuner train ./finetune/internlm2_chat_7b_qlora_interview_data.py --deepspeed deepspeed_zero2
```
3. 转换模型为hf格式：
```
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./finetune/internlm2_chat_7b_qlora_interview_data.py \
                         ./work_dirs/internlm2_chat_7b_qlora_interview_data/iter_{}.pth \
                         ./work_dirs/internlm2_chat_7b_qlora_interview_data/iter_{}_hf
```
4. 合并模型：
```
xtuner convert merge ./models/internlm2_5-chat-7b \
./work_dirs/internlm2_chat_7b_qlora_interview_data/iter_{}_hf \
./models/InternLM-Interview-Assistant --max-shard-size 2GB
```
5. Imdeploy 部署-可选
```
pip install lmdeploy
python -m lmdeploy.pytorch.chat ./models/InternLM-Interview-Assistant  \
    --max_new_tokens 256 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```
6. 进行 4bit 量化-可选
```
lmdeploy lite auto_awq ./models/InternLM-Interview-Assistant --work-dir ./models/InternLM-Interview-Assistant-4bit
```
7. 测试速度-可选
```
python ./benchmark/benchmark_transformer.py
python ./benchmark/benchmark_lmdeploy.py 
```
得到速度对比，可以看到使用 LMdeploy 的 Turbomind 和 4bit 量化模型可以明显提升推理速度。
||||
|-|-|-|
|Model|Toolkit|speed(words/s)
InternLM-Interview-Assistant|transformer|66.378
InternLM-Interview-Assistant|LMDeploy(Turbomind)|145.431
InternLM-Interview-Assistant-4bit|LMDeploy(Turbomind)|343.990

## DPO 数据集构建
- 数据格式为：
{
  "prompt": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Who won the world series in 2020?"
    },
    {
      "role": "assistant",
      "content": "The Los Angeles Dodgers won the World Series in 2020."
    },
    {
      "role": "user",
      "content": "Where was it played?"
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "The 2020 World Series was played at Globe Life Field in Arlington, Texas."
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "I don't know."
    }
  ]
}
- 详细数据见 dpo/dpo_datas.jsonl

## 后记
本项目是个人的一个学习项目，项目还未完善，在不断优化中，如有帮助，希望能被 star

当前版本进行了重大重构，老版本可查看几周前的 Commit 历史
