from ...rag.parse_knowledge import data_parsing_others
from ..server.tools.tools_prompt import transquestion_prompt_template
from ..server.internvl.internlm_server import update_db

def upload_other(data_path):   
    questions = []
    split_docs = data_parsing_others(data_path)
    for docs in split_docs:
        messages = [{'role': 'user', "content": transquestion_prompt_template.format(docs)}]
        res = chat(messages)
        questions.append(res)
        print(res)
    update_db(questions)