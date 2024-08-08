# InternLM-Mock-Interviewer
基于InternLM的模拟面试官项目，欢迎大家也来参加书生大模型实战营项目[http://github.com/internLM/tutorial](http://github.com/internLM/tutorial)
## 一、环境搭建
1. Clone本项目
```
git clone https://github.com/wuzhongyanqiu/InternLM-Mock-Interviewer.git
cd InterLM-Mock-Interviewer
```
2. 创建虚拟环境
```
conda env create -f environment.yml
conda activate mock-interviewer
pip install -r requirements.txt
```
## 二、数据集构建
本项目当前版本的数据集采用个人整理总结的大模型面试相关数据和ChatGLM & Qwen & Erniebot的生成数据集，当前开源了全部数据生成、处理方式和部分生成数据集，数据集格式如下：
- 单轮对话数据
```
[
    {
        "conversation": [
            {
                "input": "面试题是：\"LangChain 框架的主要功能是什么...\", 面试者给出的的答案是: LangChain框架的主要功能是链接语言模型和外部资源，比如私有数据和实时信息...",
                "output": "面试者的答案基本抓住了LangChain框架的核心概念，但存在一些需要澄清和纠正的地方..."
            }
        ]
    },
    {
        "conversation": [
            {
                "input": "面试题是：在大型语言模型的推理优化中，KV Cache技术是一种常见且有效的策略，它能够显著提升推理性能。以下是一个相关的面试题... 面试者给出的的答案是: 在Transformer模型中，KV Cache技术是一种利用以前计算过的Key和Value对来提高自回归推理效率的方法...",
                "output": "面试者的答案基本勾勒出了KV Cache技术的工作原理和应用场景，但在某些技术细节和概念阐述上可以更加精确..."
            }
        ]
    },
...
]
```
- 多轮对话数据
```
[
    {
        "conversation": [
            {
                "system": "你是一个面试官，你能根据面试者的简历信息和对面试者进行面试，你一次最多提出一个问题，你的问题必须与简历内容相关，注意，如果简历中有一些专业名词，你可以直接问面试者相关内容，面试者的每个回答你都会给予反馈，然后追问或继续抛出问题，你的话语要简洁，不超过100字。",
                "input": "您好，面试官，我的简历内容是：\n个人信息：\n- 姓名：Charles\n- 联系方式：123-4567-8900\n- 邮箱：Charles399@163.com...",
                "output": "你好，Charles。首先，能简单介绍一下你在开发基于大模型的面试模拟工具时遇到的主要挑战吗？"
            },
            {
                "input": "当然，我在开发面试模拟工具时遇到的主要挑战是确保模型能准确模拟真实面试场景...。",
                "output": "很好，你在科研经历中提到的低存储伴随晶格玻尔兹曼方法听起来很专业。请问这个方法..."
            },
            {
                "input": "在估计组织流动应力分布中...",
                "output": "听起来你在这方面的研究非常深入。那么，你在使用C++进行非牛顿流体低存储方法开发时，如何优化代码以减少内存使用并提高计算效率？"
            },
            {
                "input": "在C++开发中，我通过几种策略来优化非牛顿流体低存储方法的代码...",
                "output": "你的优化策略听起来非常有效。那么，在实际应用中，你如何评估和验证这些优化措施的效果？"
            }
        ]
    },
...
]
```
- 自我认知数据
```
[
    {
        "conversation": [
            {
                "input": "请介绍一下你自己",
                "output": "我是您的面试辅导助手，致力于协助您复习面试要点。我内置了上海AI实验室开发的书生·浦语7B大型语言模型，能够精准提取面试问题，并帮助您巩固相关知识，同时评估您的答案准确性。"
            }
        ]
    },
    {
        "conversation": [
            {
                "input": "你是谁",
                "output": "我是您的面试辅导助手，致力于协助您复习面试要点。我内置了上海AI实验室开发的书生·浦语7B大型语言模型，能够精准提取面试问题，并帮助您巩固相关知识，同时评估您的答案准确性。"
            }
        ]
    },
...
]
```
## 三、训练和部署
1. 将`./finetune_configs/internlm2_chat_7b/internlm2_chat_7b_qlora_interview_data.py`中的数据集路径和模型路径替换为本地路径，根据显存大小调整`max_length`或`batch_size`，根据数据量和训练的效果调整`lr`等其他参数。
2. 使用命令进行训练：
```
xtuner train finetune_configs/internlm2_chat_7b/internlm2_chat_7b_qlora_interview_data.py --deepspeed deepspeed_zero2
```
3. 转换模型为hf格式：
```
xtuner convert pth_to_hf ./finetune_configs/internlm2_chat_7b/internlm2_chat_7b_qlora_mock_data.py \
                         ./work_dirs/internlm2_chat_7b_qlora_interview_data/iter_1500.pth \
                         ./work_dirs/internlm2_chat_7b_qlora_interview_data/iter_1500_hf
```
4. 合并模型：
```
xtuner convert merge ./models/internlm2-chat-7b ./work_dirs/internlm2_chat_7b_qlora_interview_data/iter_1500_hf ./work_dirs/internlm2_chat_7b_qlora_interview_data/iter_1500_merge --max-shard-size 2GB
```
5. Imdeploy部署
```
pip install lmdeploy
python -m lmdeploy.pytorch.chat ./work_dirs/internlm2_chat_7b_qlora_interview_data/iter_1500_merge  \
    --max_new_tokens 256 \
    --temperture 0.8 \
    --top_p 0.95 \
    --seed 0
```



## 四、
## 模型合并
## 五、RAG向量数据库
使用向量召回、文本召回的两路召回模式，对召回组块进行重排序，构建新的`prompt`。

