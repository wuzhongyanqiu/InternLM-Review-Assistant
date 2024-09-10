import requests
from lagent.actions.base_action import BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode
import json

class RAG(BaseAction):

    def __init__(self):
        super().__init__()

    @tool_api
    def gen_database(self):
        """一个创建检索增强知识库的API。当你需要创建知识库时，可以使用它。
            
        """
        url = 'http://127.0.0.1:8004/rag/gendb'
        try:
            response = requests.post(url)
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_gendatabase exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {"state": "知识库创建完成"}

    @tool_api
    def get_instruction(self):
        """一个获得模拟面试指导说明的API。当你需要开启模拟面试时，首先应该使用它来获得一些模拟面试相关事项，利用它进行面试。

        Returns:
            :class:`dict`: 面试相关的事项，包括：
                * instruction (str): 模拟面试指导说明 
        """

        instruction = """
        当开始模拟面试时，你首先应当让面试者上传简历，当得到确定的回答时，使用 get_resumes 工具获得简历内容，并根据简历内容进行第一次提问。
        当面试者进行回答后，你需要根据当前对话内容判断是否要进行点评，如果需要，使用 get_comments 得到点评信息，并组织语言进行点评。
        你还需要判断是否追问，如果追问，使用 get_questions 抽取一些和当前对话内容相关的问题，并组织语言进行追问。
        当一个话题结束时，你可以再次根据简历内容进行另一个话题的提问。
        注意，模拟面试过程中你和面试者的对话应该流畅，符合真实面试场景。

        """

        return {'instruction': instruction}

    @tool_api
    def get_comments(self, query: str, ans: str) -> dict:
        """一个可以获得点评信息的API。当你需要对一个面试者的回答进行答案评估时，可以使用它。输入应该是问题和面试者的回答。

        Args:
            query (str): 问题
            ans (str): 面试者的回答
        
        Returns:
            :class:`dict`: 得到的点评信息，包括：
                * result (str): 点评结果 
        """
        url = "http://0.0.0.0:8004/rag/comments"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"query": query, "ans": ans})
        try:
            comments = requests.post(url, data=payload, headers=headers).json()
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_getcomments exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {'comments': comments['comments']}

    @tool_api
    def get_questions(self, chat_content: str) -> dict:
        """一个可以从题库抽取相关问题的API，当你需要抽取一些问题时，可以使用它。
        
        Args:
            chat_content (str): 当前对话的内容

        Returns:
            :class:`dict`: 抽取到的问题，包括：
                * question (str): 问题
        """
        url = "http://0.0.0.0:8004/rag/questions"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"chat_content": chat_content})
        try:
            questions = requests.post(url, data=payload, headers=headers).json()
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_getquestions exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {'question': question['question']}

    @tool_api
    def get_resumes(self) -> dict:
        """一个可以获得已上传的简历信息的API，当你确认一个简历已经被上传时，可以使用它获得简历内容。

        Returns:
            :class:`dict`: 得到的简历内容，包括：
                * resumes_content (str): 简历内容
        """
        url = "http://0.0.0.0:8004/rag/resumes"
        try:
            resumes_content = requests.post(url)
        except Exception as exc:
            return ActionReturn(
                errmsg=f'rag_getresumes exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        return {'resumes_content': resumes_content['resumes_content']}


    # @tool_api
    # def get_state(self, chat_content) -> dict:
    #     """在模拟面试过程中，这个API用于获得当前对话的状态，帮助选择辅助对话的工具，

    #     """