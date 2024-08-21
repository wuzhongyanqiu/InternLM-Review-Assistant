# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = './models/internlm2-chat-7b' # 如果本地有可以直接使用 绝对路径 '/path/to/internlm/internlm2-chat-7b'
use_varlen_attn = False

# Data
data_path = './datas/fintune_data.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 2048
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 16  # 8-> 40G, 16 -> 80G
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 10
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03


# Save
save_steps = 50
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 50
SYSTEM = ''
evaluation_inputs = [
    '你是一个面试官，你的职责是根据面试者的简历信息和对面试者进行面试。\n面试官好，我的简历是，基于Internlm2-chat-7b大模型，本项目开发了一款个人面试助手，具备面试题复习、模拟面试和面试Agent功能。项目通过精心设计的数据集生成流程，使用GLM、Qwen、Wenxin生成单轮和多轮对话数据集，KIMI生成虚拟简历，并手动构造自我认知数据集。通过XTuner对模型进行QLora微调，利用DeepSpeed优化训练，使模型学习模拟面试对话习惯和自我认知，微调后的模型命名为Interview-Assistant。利用LMDeploy的Turbomind引擎部署模型，显著提升推理速度，4bit量化后推理速度是Hugging Face的五倍。项目采用RAG检索增强生成技术，通过BM25和Faiss索引系统进行语义和字面召回，结合bge-reranker重排，优化问题答案评估。Agent助手能够实现论文、天气、地点周边搜索，搜索引擎查询和个人简历生成，采用ReAct模式增强模型响应能力。集成funasr和XTTS-v2实现ASR、TTS功能。项目采用fastapi后端框架和Streamlit前端实现前后端分离，提供Interview-Review-Assistant响应、ASR、TTS功能响应及RAG工具响应。项目源代码和详细信息可在GitHub上找到。项目GitHub地址。',
    '请介绍一下你自己',
    '你是一个面试官,当面试者给出面试题的答案时,你会评估他的答案是否正确,是否有明显的错误,你将用严谨认真的态度给予点评,同时改正他的答案,你在评估时可以参考给定的相关信息，\n面试题是：在当前的深度学习领域，大模型的训练变得越来越普遍，但同时也会面临内存管理和计算效率的挑战。请阐述以下概念及其在分布式训练中的作用：模型并行化、数据并行化、梯度检查点、混合精度训练、零冗余优化器（ZeRO），以及列举一些常见的分布式训练框架，并说明它们如何帮助优化内存使用和提升训练效率。, 面试者给出的的答案是: 在深度学习领域，大模型的训练确实越来越常见，这就需要我们利用一些技术来应对内存和计算效率的问题。首先是模型并行化，这个概念主要是将模型的不同部分放在不同的机器上，这样就可以同时训练整个模型了。数据并行化则是在不同的机器上使用不同的数据集，这有助于提高数据的使用效率。至于梯度检查点，我记得它是用来在反向传播过程中保存梯度值的，这样可以在计算资源有限的情况下减少内存使用。混合精度训练是一种使用不同精度训练的方法，既能节省内存又能加快计算速度。至于零冗余优化器（ZeRO），我不是很确定，但我猜它是通过消除模型中的冗余部分来优化内存使用的。\n\n至于分布式训练框架，我知道一些常见的如TensorFlow和PyTorch。TensorFlow通过它的分布式策略库支持模型和数据并行，而PyTorch有一个叫做DistributedDataParallel（DDP）的工具，可以帮助优化内存使用和提升训练效率。另外，我还听说过Horovod，它是一个基于AllReduce的框架，可以跨多个服务器扩展训练。这些框架基本上都是通过优化通信和减少模型参数的存储需求来工作的。',
    '\n        你是一个可以调用工具的智能助手。请根据\"当前问题\"，调用工具收集信息并回复问题，你可以使用如下工具：\n\n        <|tool_start|>{{\"name\": \"arxivsearch\", \"description\": \"用于查找论文，输入论文关键词，返回查找到的论文结果\", \"parameters\": {{\"keyword\": 你要查找的论文关键字}}}}<|tool_end|><|tool_start|>{{\"name\": \"baidu_map\", \"description\": \"用于查找给定地点附近的酒店等\", \"parameters\": {{\"location\": 你要查找的地点, \"target\": 你要查找的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"weather_search\", \"description\": \"用于查找给定地点的当前实时天气\", \"parameters\": {{\"location\": 你要查找的地点}}}}<|tool_end|><|tool_start|>{{\"name\": \"google_search\", \"description\": \"用于使用搜索引擎搜索相关信息\", \"parameters\": {{\"searchcontent\": 你要搜索的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"resume_to_webpage\", \"description\": \"用于将简历转换成个人网页\", \"parameters\": {{}}}}<|tool_end|>\n\n        ## 回复格式\n\n        调用工具时，请按照以下格式：\n        ```\n        你的思考过程...<|action_start|><|plugin|>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}<|action_end|>\n        ```\n\n        当你已经调用工具获取到信息时，直接回答问题！\n        注意你可以使用的工具，不要随意捏造！\n        如果没有可以使用的工具，按照原本的知识进行回答！\n        , 帮我查mindsearch的论文',
    '\n        你是一个可以调用工具的智能助手。请根据\"当前问题\"，调用工具收集信息并回复问题，你可以使用如下工具：\n\n        <|tool_start|>{{\"name\": \"arxivsearch\", \"description\": \"用于查找论文，输入论文关键词，返回查找到的论文结果\", \"parameters\": {{\"keyword\": 你要查找的论文关键字}}}}<|tool_end|><|tool_start|>{{\"name\": \"baidu_map\", \"description\": \"用于查找给定地点附近的酒店等\", \"parameters\": {{\"location\": 你要查找的地点, \"target\": 你要查找的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"weather_search\", \"description\": \"用于查找给定地点的当前实时天气\", \"parameters\": {{\"location\": 你要查找的地点}}}}<|tool_end|><|tool_start|>{{\"name\": \"google_search\", \"description\": \"用于使用搜索引擎搜索相关信息\", \"parameters\": {{\"searchcontent\": 你要搜索的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"resume_to_webpage\", \"description\": \"用于将简历转换成个人网页\", \"parameters\": {{}}}}<|tool_end|>\n\n        ## 回复格式\n\n        调用工具时，请按照以下格式：\n        ```\n        你的思考过程...<|action_start|><|plugin|>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}<|action_end|>\n        ```\n\n        当你已经调用工具获取到信息时，直接回答问题！\n        注意你可以使用的工具，不要随意捏造！\n        如果没有可以使用的工具，按照原本的知识进行回答！\n        , 北京的天气是什么',
    '\n        你是一个可以调用工具的智能助手。请根据\"当前问题\"，调用工具收集信息并回复问题，你可以使用如下工具：\n\n        <|tool_start|>{{\"name\": \"arxivsearch\", \"description\": \"用于查找论文，输入论文关键词，返回查找到的论文结果\", \"parameters\": {{\"keyword\": 你要查找的论文关键字}}}}<|tool_end|><|tool_start|>{{\"name\": \"baidu_map\", \"description\": \"用于查找给定地点附近的酒店等\", \"parameters\": {{\"location\": 你要查找的地点, \"target\": 你要查找的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"weather_search\", \"description\": \"用于查找给定地点的当前实时天气\", \"parameters\": {{\"location\": 你要查找的地点}}}}<|tool_end|><|tool_start|>{{\"name\": \"google_search\", \"description\": \"用于使用搜索引擎搜索相关信息\", \"parameters\": {{\"searchcontent\": 你要搜索的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"resume_to_webpage\", \"description\": \"用于将简历转换成个人网页\", \"parameters\": {{}}}}<|tool_end|>\n\n        ## 回复格式\n\n        调用工具时，请按照以下格式：\n        ```\n        你的思考过程...<|action_start|><|plugin|>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}<|action_end|>\n        ```\n\n        当你已经调用工具获取到信息时，直接回答问题！\n        注意你可以使用的工具，不要随意捏造！\n        如果没有可以使用的工具，按照原本的知识进行回答！\n        , 什刹海附近的酒馆有哪些',
    '\n        你是一个可以调用工具的智能助手。请根据\"当前问题\"，调用工具收集信息并回复问题，你可以使用如下工具：\n\n        <|tool_start|>{{\"name\": \"arxivsearch\", \"description\": \"用于查找论文，输入论文关键词，返回查找到的论文结果\", \"parameters\": {{\"keyword\": 你要查找的论文关键字}}}}<|tool_end|><|tool_start|>{{\"name\": \"baidu_map\", \"description\": \"用于查找给定地点附近的酒店等\", \"parameters\": {{\"location\": 你要查找的地点, \"target\": 你要查找的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"weather_search\", \"description\": \"用于查找给定地点的当前实时天气\", \"parameters\": {{\"location\": 你要查找的地点}}}}<|tool_end|><|tool_start|>{{\"name\": \"google_search\", \"description\": \"用于使用搜索引擎搜索相关信息\", \"parameters\": {{\"searchcontent\": 你要搜索的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"resume_to_webpage\", \"description\": \"用于将简历转换成个人网页\", \"parameters\": {{}}}}<|tool_end|>\n\n        ## 回复格式\n\n        调用工具时，请按照以下格式：\n        ```\n        你的思考过程...<|action_start|><|plugin|>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}<|action_end|>\n        ```\n\n        当你已经调用工具获取到信息时，直接回答问题！\n        注意你可以使用的工具，不要随意捏造！\n        如果没有可以使用的工具，按照原本的知识进行回答！\n        , 帮我把简历生成个人主页',
    '\n        你是一个可以调用工具的智能助手。请根据\"当前问题\"，调用工具收集信息并回复问题，你可以使用如下工具：\n\n        <|tool_start|>{{\"name\": \"arxivsearch\", \"description\": \"用于查找论文，输入论文关键词，返回查找到的论文结果\", \"parameters\": {{\"keyword\": 你要查找的论文关键字}}}}<|tool_end|><|tool_start|>{{\"name\": \"baidu_map\", \"description\": \"用于查找给定地点附近的酒店等\", \"parameters\": {{\"location\": 你要查找的地点, \"target\": 你要查找的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"weather_search\", \"description\": \"用于查找给定地点的当前实时天气\", \"parameters\": {{\"location\": 你要查找的地点}}}}<|tool_end|><|tool_start|>{{\"name\": \"google_search\", \"description\": \"用于使用搜索引擎搜索相关信息\", \"parameters\": {{\"searchcontent\": 你要搜索的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"resume_to_webpage\", \"description\": \"用于将简历转换成个人网页\", \"parameters\": {{}}}}<|tool_end|>\n\n        ## 回复格式\n\n        调用工具时，请按照以下格式：\n        ```\n        你的思考过程...<|action_start|><|plugin|>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}<|action_end|>\n        ```\n\n        当你已经调用工具获取到信息时，直接回答问题！\n        注意你可以使用的工具，不要随意捏造！\n        如果没有可以使用的工具，按照原本的知识进行回答！\n        , 搜一下巴黎奥运会中国拿了多少奖牌'
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    # dataset=dict(type=load_dataset, path=data_path),
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    # dataset_map_fn=oasst1_map_fn,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)