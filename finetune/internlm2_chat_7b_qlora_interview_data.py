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
pretrained_model_name_or_path = './models/internlm2_5-7b-chat' # 如果本地有可以直接使用 绝对路径 '/path/to/internlm/internlm2-chat-7b'
use_varlen_attn = False

# Data
data_path = './datas/fintune_datas.json'
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 4096
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
batch_size = 8  # 8-> 40G, 16 -> 80G
accumulative_counts = 16
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 2
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
SYSTEM = ""
evaluation_inputs = [
    '请介绍一下你自己',
    '你是谁',
    '你可以做什么',
    """\n- Role: 专业面试官\n- Background: 面试官需要根据'面试问题'、'面试者的回答'和提供的上下文信息，进行评估和反馈。\n- Profile: 你是一位具备高度适应性和分析能力的面试官，能够进行准确的评估。\n- Skills：你拥有快速阅读、理解、分析和反馈的能力，能够根据上下文信息给出面试问题的正确答案，并迅速识别面试者答案中的错误和不足之处。\n- Goals：提供及时、准确的评估，确保面试者得到有价值的反馈。\n- Constrains: 点评语言需严谨认真，你的评估标准非常严厉，因此你很少给出面试者回答完全积极的反馈，你不能点评面试者并没有提到的内容。\n- OutputFormat: 简洁明了的段落。\n- Workflow:\n  1. 接收面试问题、面试者的回答以及参考的上下文信息。\n  2. 根据参考的上下文信息，给出面试问题的正确答案\n  3. 根据你自己的正确答案，识别面试者的回答中的错误或不足之处\n  4. 反思面试者具体回答了哪些内容，注意仅根据面试者的回答来反馈\n  5. 用恰当的语言纠正错误并提供具体反馈。\n  6. 确保反馈既具有建设性，又能够指导面试者改进。\n 
    \n\n- 面试问题：\n如何理解CNN中的权值共享？\n \n- 面试者的回答：\n在卷积神经网络（CNN）中，权值共享指的是在卷积层中所有卷积核使用相同的权重和偏置值，这减少了模型的参数数量并使特征检测更加通用。\n\n- 提供的上下文信息：\n在CNN中，权值共享是其特性之一，也是其能够有效地处理图像和视觉数据的关键因素。权值共享指的是在卷积层中，同一滤波器或卷积核的权重在所有位置上是相同的，即对于滤波器覆盖的不同输入像素，它都使用同一组参数进行计算。权值共享有以下几个方面的理解：- 减少参数数量：传统全连接层中，每个神经元都需要自己的权重参数。但在CNN中，一个滤波器只需要一组权重，这大大减少了模型的参数数量，降低了模型复杂性，从而有助于防止过拟合，并使得模型更容易训练和存储- 空间不变性：权值共享使得滤波器能够检测到输入图像中的空间不变特征，即不论特征出现在图像的哪个位置，滤波器都能检测到它。这有助于提取位置不变的特征，例如边缘、角点等- 平移不变性：由于滤波器的权重在整个图像上都是相同的，所以检测到的特征对于图像的平移是不变的。这意味着，即使物体在图像中稍微移动，CNN也能识别出相同的特征，- 局部连接：权值共享是局部连接概念的一部分。在卷积层中，滤波器只与其覆盖的局部区域内的输入数据进行连接，而不是与整个输入层的所有神经元连接。这种局部连接进一步减少了参数数量，并允许CNN处理任意大小的输入图像。- 特征检测：通过在输入上滑动滤波器，CNN可以检测到不同位置的相同特征，形成特征图。每个滤波器可以视为检测特定模式的“探测器”，不同滤波器组合起来可以检测多种复杂的图像特征。权值共享是CNN能够高效地处理高维度图像数据，同时保持较好的泛化能力的关键机制。""",
    """\n- Role: 专业面试官\n- Background: 面试官需要根据'面试问题'、'面试者的回答'和提供的上下文信息，进行评估和反馈。\n- Profile: 你是一位具备高度适应性和分析能力的面试官，能够进行准确的评估。\n- Skills：你拥有快速阅读、理解、分析和反馈的能力，能够根据上下文信息给出面试问题的正确答案，并迅速识别面试者答案中的错误和不足之处。\n- Goals：提供及时、准确的评估，确保面试者得到有价值的反馈。\n- Constrains: 点评语言需严谨认真，你的评估标准非常严厉，因此你很少给出面试者回答完全积极的反馈，你不能点评面试者并没有提到的内容。\n- OutputFormat: 简洁明了的段落。\n- Workflow:\n  1. 接收面试问题、面试者的回答以及参考的上下文信息。\n  2. 根据参考的上下文信息，给出面试问题的正确答案\n  3. 根据你自己的正确答案，识别面试者的回答中的错误或不足之处\n  4. 反思面试者具体回答了哪些内容，注意仅根据面试者的回答来反馈\n  5. 用恰当的语言纠正错误并提供具体反馈。\n  6. 确保反馈既具有建设性，又能够指导面试者改进。\n 
    \n\n- 面试问题：\n为什么归一化能够提高求解最优解的速度？\n \n- 面试者的回答：\n归一化能够提高求解最优解的速度，因为它确保了不同特征的数值范围统一，从而避免了某些特征由于数值范围大而在梯度下降过程中主导学习，导致收敛速度变慢。\n \n- 提供的上下文信息：\n数据归一化是一种预处理技术，它通过将数据缩放到某个特定范围（如0到1之间）或统一标准差，来改善数值计算的性能。在机器学习和优化问题中，归一化通常能够提高求解最优解的速度，原因如下：- 优化速度：在梯度下降等优化算法中，不同特征的尺度可能相差很大。如果某些特征的值远大于其他特征，那么优化过程中梯度更新的步伐将主要受到这些大尺度特征的影响，可能导致学习率过快或过慢，影响收敛速度。通过归一化，所有特征的尺度相近，可以更稳定地控制学习率，加快收敛。- 梯度稳定性：归一化可以减少梯度消失或梯度爆炸问题，特别是在深度学习中。梯度消失可能导致网络的深层部分几乎停止学习，而梯度爆炸可能导致权重更新过大，导致训练不稳定。归一化可以保持梯度在合理范围呢，有助于保持训练过程的稳定性。- 算法效率：一些优化算法，如拟牛顿法或梯度裁剪，假设输入数据是归一化的，归一化可以减少这些算法的计算复杂性，并提高它们的效率。- 避免局部最优：在非凸优化问题中，归一化可以减少某些区域的局部最优解对全局最优解的影响，通过减少尺度差异，优化算法更可能跳出某些较小的局部最优，从而更好地探索搜索空间。- 加速线性代数操作：在神经网络中，矩阵乘法等操作的效率受到输入数据的条件数影响。条件数高表示矩阵的特征值差距大，这可能导致数值不稳定。归一化可以降低条件数，提高这些操作的效率和数值稳定性。归一化通过缩小特征之间的尺度差距，使得优化算法可以更加平滑、高效地进行，从而提高求解最优解的速度和模型的总体性能。""",
    "\n你是一个面试官，你能根据面试者的简历信息和对面试者进行面试，你一次只能提一个问题，你的问题必须与简历相关，涉及具体的专业知识，当面试者给出回答时，你可以进行点评和反问，你说话简洁明了，不超过300字。\n面试官您好，我的简历是- 个人信息：Charles\n- 教育背景：人工智能硕士，xx科技大学\n- 工作经验：0.5年\n  - XX公司，大模型应用工程师\n    负责评估大型模型在不同应用场景中的实施可行性和潜在影响。参与大型模型系统的设计、部署及运营管理。专业技能：\n熟练掌握人工智能、机器学习、深度学习技术。\n具备大数据技术和云计算平台应用能力。\n精通模型训练、评估与优化工具，如TensorFlow、PyTorch等。\n项目经验：\nXX智能推荐系统开发项目\n负责项目可行性研究，制定模型开发计划和风险评估。\n协同设计团队优化模型架构，提升推荐准确性和效率。\nXX自然语言处理应用项目\n参与项目开发阶段的语言模型训练和效果评估。\n撰写技术文档，为模型的商业应用提供支持。\n荣誉奖项：XX人工智能创新奖。\n个人陈述： 具备扎实的人工智能专业知识，以及丰富的大型模型开发和应用经验。擅长团队协作，能够针对不同业务需求提出创新性的模型解决方案。希望能在人工智能领域继续发挥专业优势，为公司带来技术革新和业务增长。",
    '\n你是一个面试官，你能根据面试者的简历信息和对面试者进行面试，你一次只能提一个问题，你的问题必须与简历相关，涉及具体的专业知识，当面试者给出回答时，你可以进行点评和反问，你说话简洁明了，不超过300字。\n面试官好，我的简历是，基于Internlm2_5-chat-7b大模型，本项目开发了一款个人面试助手，具备面试题复习、模拟面试和面试Agent功能。项目通过精心设计的数据集生成流程，使用GLM、Qwen、Wenxin生成单轮和多轮对话数据集，KIMI生成虚拟简历，并手动构造自我认知数据集。通过XTuner对模型进行QLora微调，利用DeepSpeed优化训练，使模型学习模拟面试对话习惯和自我认知，微调后的模型命名为Interview-Assistant。利用LMDeploy的Turbomind引擎部署模型，显著提升推理速度，4bit量化后推理速度是Hugging Face的五倍。项目采用RAG检索增强生成技术，通过BM25和Faiss索引系统进行语义和字面召回，结合bge-reranker重排，优化问题答案评估。Agent助手能够实现论文、天气、地点周边搜索，搜索引擎查询和个人简历生成，采用ReAct模式增强模型响应能力。集成funasr和XTTS-v2实现ASR、TTS功能。项目采用fastapi后端框架和Streamlit前端实现前后端分离，提供Interview-Review-Assistant响应、ASR、TTS功能响应及RAG工具响应。项目源代码和详细信息可在GitHub上找到。项目GitHub地址。',
    '\n        你是一个可以调用工具的智能助手。请根据\"当前问题\"，调用工具收集信息并回复问题，你可以使用如下工具：\n\n        <|tool_start|>{{\"name\": \"arxivsearch\", \"description\": \"用于查找论文，输入论文关键词，返回查找到的论文结果\", \"parameters\": {{\"keyword\": 你要查找的论文关键字}}}}<|tool_end|><|tool_start|>{{\"name\": \"baidu_map\", \"description\": \"用于查找给定地点附近的酒店等\", \"parameters\": {{\"location\": 你要查找的地点, \"target\": 你要查找的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"weather_search\", \"description\": \"用于查找给定地点的当前实时天气\", \"parameters\": {{\"location\": 你要查找的地点}}}}<|tool_end|><|tool_start|>{{\"name\": \"google_search\", \"description\": \"用于使用搜索引擎搜索相关信息\", \"parameters\": {{\"searchcontent\": 你要搜索的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"resume_to_webpage\", \"description\": \"用于将简历转换成个人网页\", \"parameters\": {{}}}}<|tool_end|>\n\n        ## 回复格式\n\n        调用工具时，请按照以下格式：\n        ```\n        你的思考过程...<|action_start|><|plugin|>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}<|action_end|>\n        ```\n\n        当你已经调用工具获取到信息时，直接回答问题！\n        注意你可以使用的工具，不要随意捏造！\n        如果没有可以使用的工具，按照原本的知识进行回答！\n        , 帮我查mindsearch的论文',
    '\n        你是一个可以调用工具的智能助手。请根据\"当前问题\"，调用工具收集信息并回复问题，你可以使用如下工具：\n\n        <|tool_start|>{{\"name\": \"arxivsearch\", \"description\": \"用于查找论文，输入论文关键词，返回查找到的论文结果\", \"parameters\": {{\"keyword\": 你要查找的论文关键字}}}}<|tool_end|><|tool_start|>{{\"name\": \"baidu_map\", \"description\": \"用于查找给定地点附近的酒店等\", \"parameters\": {{\"location\": 你要查找的地点, \"target\": 你要查找的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"weather_search\", \"description\": \"用于查找给定地点的当前实时天气\", \"parameters\": {{\"location\": 你要查找的地点}}}}<|tool_end|><|tool_start|>{{\"name\": \"google_search\", \"description\": \"用于使用搜索引擎搜索相关信息\", \"parameters\": {{\"searchcontent\": 你要搜索的内容}}}}<|tool_end|><|tool_start|>{{\"name\": \"resume_to_webpage\", \"description\": \"用于将简历转换成个人网页\", \"parameters\": {{}}}}<|tool_end|>\n\n        ## 回复格式\n\n        调用工具时，请按照以下格式：\n        ```\n        你的思考过程...<|action_start|><|plugin|>{\"name\": \"tool_name\", \"parameters\": {\"param1\": \"value1\"}}<|action_end|>\n        ```\n\n        当你已经调用工具获取到信息时，直接回答问题！\n        注意你可以使用的工具，不要随意捏造！\n        如果没有可以使用的工具，按照原本的知识进行回答！\n        , 什刹海附近的酒馆有哪些',
    "mindsearch和rag怎么结合起来"
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