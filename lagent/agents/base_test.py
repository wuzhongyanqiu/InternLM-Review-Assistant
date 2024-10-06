question = """
什么是猜测推理技术？
"""

answer = """
猜测推理技术是先进行猜测推理，然后再生成的技术，具体细节我并不太了解
"""

content = """
也称为猜测抽样、辅助生成或分块并行解码，猜测推理是一种不同的并行执行LLM的方法。通常，GPT-style大型语言模型是自回归模型，逐个令牌地生成文本。

生成的每个令牌都依赖于之前所有的令牌来提供上下文。这意味着在常规执行中，不可能并行地从同一个序列中生成多个令牌-你必须等待第n个令牌生成后才能生成n+1个。

下图显示了猜测推理的一个例子，其中草案模型暂时预测了多个未来步骤，并且可以并行验证或拒绝。在这种情况下，草案中的前两个预测令牌被接受，而最后一个在继续生成之前被拒绝并删除。

猜测抽样提供了一种解决办法。这种方法的基本思想是使用一些”更便宜“的过程生成几个令牌长的草稿连续输出。然后，在所需的执行步骤中使用廉价的草稿作为”猜测“上下文，同时并行地执行主要的”验证“模型。

如果验证模型生成与草稿相同的令牌，则可以确定接受这些令牌作为输出。否则，可以丢弃第一个不匹配令牌之后的所有内容，并使用新的草稿重复该过程。

有许多不同的选项可以用来生成草稿令牌，并且每种方法都有不同的权衡。可以训练多个模型或在单个预训练模型上微调多个头部，以预测多个步骤的令牌。或者可以使用一个小模型作为草稿模型，并使用一个更大、更能胜任地模型作为验证器。
"""

messages = [
    {"role": "system", "content": "根据你已有的知识和上下文内容，回答问题，要求语言简洁通顺，答案准确无误"},
    {"role": "user", "content": "问题：\n{question}\n上下文内容：\n{content}\n".format(question=question, content=content)}
]

from lmdeploy.serve.openai.api_client import APIClient
api_client = APIClient('http://127.0.0.1:23333')
model_name = api_client.available_models[0]

for item in api_client.chat_completions_v1(model=model_name, messages=messages):
    pass

res = item

rightans = res['choices'][0]['message']['content']
messages_ = [
    {"role": "system", "content": "根据你已有的知识和正确答案，对面试者的答案进行点评"},
    {"role": "user", "content": "正确答案：\n{rightans}\n面试者的答案: \n{ans}\n".format(rightans=rightans, ans=answer)}
] 

for item in api_client.chat_completions_v1(model=model_name, messages=messages_):
    pass

res = item

final_res = res['choices'][0]['message']['content']
print(final_res)


messages = [
    {"role": "system", "content": "你是一个面试官，你将根据对话内容生成追问标记或者是提问标记"},
    {"role": "user", "content": "问题：\n{question}\n上下文内容：\n{content}\n".format(question=question, content=content)}
]

