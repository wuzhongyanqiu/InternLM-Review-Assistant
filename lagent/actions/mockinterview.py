import requests
from lagent.actions.base_action import BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
RESUME_PATH = os.path.join(current_dir, "../../tmp_dir/resume/resume.pdf")

class MockInterview(BaseAction):

    def __init__(self):
        super().__init__()

    @tool_api
    def get_comments(self, mockinterview_query: str, mockinterview_ans: str) -> dict:
        """在模拟面试期间，当面试者刚刚回答了一个技术问题，面试官想对其回答的深度和准确性进行评价时，可以使用这个 API 得到参考信息。
           在模拟面试期间，当面试者提供了很好的解决方案，面试官希望给予正面的反馈并建议一些提升点时。可以使用这个 API 得到需要的参考信息。

        Args:
            mockinterview_query (str): 问题
            mockinterview_ans (str): 面试者的回答
        
        Returns:
            :class:`dict`: 得到的相关信息，包括：
                * result (str): 相关信息 
        """
        url = "http://0.0.0.0:8004/rag/mockinterview_comments"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"mockinterview_query": mockinterview_query, "mockinterview_ans": mockinterview_ans})
        try:
            comments = requests.post(url, data=payload, headers=headers).json()
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_getcomments exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {'type': 'text', 'content': comments['comments']}

    @tool_api
    def get_questions(self, mockinterview_keywords: str) -> dict:
        """在模拟面试期间，当面试者对某个技术领域的回答不够深入，面试官决定根据对话内容选择一些逗号分隔的关键词作为参数，从题库中中抽取一些相关的问题，以进一步评估面试者的知识时，可以使用这个 API 得到需要的相关问题。
           在模拟面试期间，当一个话题结束，面试官希望根据技术栈出一道专业问题来对面试者提问时，可以使用这个 API 得到需要的相关问题。
        
        Args:
            mockinterview_keywords (str): 要进行题库查找的关键词，用逗号分隔

        Returns:
            :class:`dict`: 抽取到的问题，包括：
                * question (str): 问题
        """
        url = "http://0.0.0.0:8004/rag/mockinterview_questions"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"mockinterview_keywords": mockinterview_keywords})
        try:
            questions = requests.post(url, data=payload, headers=headers).json()
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_getquestions exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {'type': 'text', 'content': questions['question']}

    @tool_api
    def get_resumes(self) -> dict:
        """在模拟面试期间，面试官发现有提到的项目或技能需要进一步探讨，可以获取简历了解面试者项目或技能的细节以便提问时，可以使用这个 API 得到简历信息。
           在模拟面试期间，在面试过程中，面试官希望得知面试者的技术栈时，可以使用这个 API 得到简历信息。

        Returns:
            :class:`dict`: 得到的简历内容，包括：
                * resumes_content (str): 简历内容
        """
        url = "http://0.0.0.0:8004/rag/mockinterview_resumes"

        try:
            resumes_content = requests.post(url).json()
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_getresumes exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {'type': 'text', 'content': resumes_content['resumes_content']}



