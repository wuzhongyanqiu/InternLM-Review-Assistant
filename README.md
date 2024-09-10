# InternLM-Interview-Assistant
基于 InternLM 的面试复习助手项目，欢迎大家也来参加书生大模型实战营项目[http://github.com/internLM/tutorial](http://github.com/internLM/tutorial)

## 简介
学习时，往往整理了很多知识资料，但苦于没有一个伙伴帮助查漏补缺，也没有现成的对口合适的题目用于检验学习效果，因此，面试复习助手就是为了解决这个难题，只需要将知识资料一股脑上传，助手会将其解析，为你开启定制化的模拟面试。

## 项目功能
用户上传个人文件，面试助手解析文件，将文本、表格、公式、图片分割成有完整语义的段落，建立知识库和面试题库，在模拟面试过程中，面试助手将利用 RAG 进行关键词抽题、答案评估、以文搜图等功能，支持语音输入和输出。

## 架构图

![架构图](./assets/architecture_diagram.png)

## 演示视频

## 快速开始
- 1. clone 本项目至本地开发机
```bash
git clone https://github.com/wuzhongyanqiu/InternLM-Interview-Assistant.git
```

- 2. 配置环境
```
# 创建虚拟环境
conda create -n interview-assistant python=3.10
# 激活环境
conda activate interview-assistant
# 安装所需依赖
cd InternLM-Interview-Assistant
pip install -r requirements.txt
```

- 3. 启动
```
python app.py
```

- 4. 示例效果

## 思路讲解视频


## 后记
本项目是个人的一个学习项目，项目还未完善，在不断优化中，如有帮助，希望能被 star
当前版本进行了重大重构，老版本可查看几周前的 Commit 历史
