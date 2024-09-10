system_prompt = "<s><|im_start|>system\n{system}<|im_end|>\n"
user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
assistant_prompt = '<|im_start|>assistant\n{assistant}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'

def convert_prompt(messages):
    total_prompt = ''
    for message in messages[:-1]:
        cur_content = message['content']
        if message['role'] == 'system':
            cur_prompt = system_prompt.format(system=cur_content)
        elif message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'assistant':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=messages[-1]['content'])

    return total_prompt

