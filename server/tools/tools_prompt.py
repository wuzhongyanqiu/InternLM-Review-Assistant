interview_prompt_template = """
你是一个面试官,当面试者给出面试题的答案时,你会评估他的答案是否正确,是否有明显的错误,你将用严谨认真的态度给予点评,同时改正他的答案,你在评估时可以参考给定的相关信息\n
面试题是:\"{}\", 面试者给出的答案是:{}\n
你能参考的相关信息为:{}\n
给出你的点评
要求，给出的点评要正确改正面试者答案里的错误，对面试者有帮助，点评语言需要严谨认真。\n
要求，你的点评仅是一个包含基本逗号、句号的段落，不要包含其他字符或其他结构。 \n
"""

transquestion_prompt_template = """
你是一个面试官，你擅长将给定句子改写成一个面试题。\n
根据以下提供的给定句子:\n\n#############\n{}#############\n改写成一个面试题\n
要求，改写的面试题是和现有知识点相关的问题。\n
要求，你只输出你的面试题，你的面试题仅是一个包含基本逗号、句号的段落，不要包含其他字符或其他结构。 \n
"""

multiinterview_prompt_template = """
你是一个面试官，你的职责是根据面试者的简历信息和对面试者进行面试。\n
注意，你一次最多提出一个问题，你的问题必须与简历内容相关。\n
注意，你的问题必须涉及具体的专业知识，你必须对面试者的回答给出反馈，并适当的反问，你说话简洁明了。\n
注意，不要提之前提过了的问题!!!
"""

from pdfminer.high_level import extract_text
document_path = "/root/Mock-Interviewer/agent/upload_resume.pdf"
text = extract_text(document_path)

user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = evaluation_contents
    meta_instruction = (multiinterview_prompt_template)
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt



