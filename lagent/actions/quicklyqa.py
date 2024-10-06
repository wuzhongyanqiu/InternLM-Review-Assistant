import requests
from lagent.actions.base_action import BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
RESUME_PATH = os.path.join(current_dir, "../../tmp_dir/resume/resume.pdf")

class QuicklyQA(BaseAction):

    def __init__(self):
        super().__init__()

    @tool_api
    def get_query(self) -> dict:
        """在快问快答期间，当轮到你提问时，使用这个 API 从题库中抽取一道题提问。
           在快问快答期间，当用户回答完问题，而且你对其回答进行评估以后，使用这个 API 再次进行提问。
        
        Returns:
            :class:`dict`: 抽取到的题目及其编号，包括：
                * result (str): 题目内容和编号
        """
        url = "http://0.0.0.0:8004/rag/quicklyQA_questions"

        try:
            query = requests.post(url).json()
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_getcomments exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {'type': 'text', 'content': query['query']}

    @tool_api
    def get_comments(self, quicklyQA_query: int, quicklyQA_ans: str) -> dict:
        """在快问快答期间，当用户回答了你提出的问题后，使用这个 API 得到评估的信息。

        Args:
            quicklyQA_querymum (int): 问题编号
            quicklyQA_ans (str): 面试者的回答
        
        Returns:
            :class:`dict`: 得到的专业评估信息，包括：
                * result (str): 评估信息内容 
        """
        url = "http://0.0.0.0:8004/rag/quicklyQA_comments"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"quicklyQA_query": quicklyQA_query, "quicklyQA_ans": quicklyQA_ans})
        try:
            comments = requests.post(url, data=payload, headers=headers).json()
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_getcomments exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {'type': 'text', 'content': comments['comments']}


