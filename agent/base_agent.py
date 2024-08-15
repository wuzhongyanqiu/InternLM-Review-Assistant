from agent.baidu_map import BaiduMap
from agent.arxiv_search import ArxivSearch
from agent.google_search import GoogleSearch
from agent.weather_search import WeatherSearch
from agent.resume2webpage import RESUME2WEBSITE
import re

class BaseAgent:
    def __init__(self):
        self.system_prompt = """
        你是一个可以调用工具的智能助手。请根据"当前问题"，调用工具收集信息并回复问题，你可以使用如下工具：\n
        {prompt}\n
        ## 回复格式

        调用工具时，请按照以下格式：
        ```
        你的思考过程...<|action_start|><|plugin|>{{"name": "tool_name", "parameters": {{"param1": "value1"}}}}<|action_end|>
        ```

        当你已经调用工具获取到信息时，直接回答问题！
        注意你可以使用的工具，不要随意捏造！
        如果没有可以使用的工具，按照原本的知识进行回答！
        """
        self.action_prompt="""
        你是一个根据你自己已经获取到的内容回答"当前问题"的助手，你已经获取到的的内容是: \n
        {context}\n
        ## 回复格式

        回复问题时，请按照以下格式：
        ```
        你的思考过程...
        ```
        注意：当你得到html和css代码后，将优化后的代码回复
        """
        self.tools = {}
        self.tools_prompt = ''
        self.input_tools()

    def input_tools(self):
        self.tools['arxivsearch'] = """<|tool_start|>{{"name": "arxivsearch", "description": "用于查找论文，输入论文关键词，返回查找到的论文结果", "parameters": {{"keyword": 你要查找的论文关键字}}}}<|tool_end|>"""
        self.tools['baidumap'] = """<|tool_start|>{{"name": "baidu_map", "description": "用于查找给定地点附近的酒店等", "parameters": {{"location": 你要查找的地点, "target": 你要查找的内容}}}}<|tool_end|>"""
        self.tools['weathersearch'] = """<|tool_start|>{{"name": "weather_search", "description": "用于查找给定地点的当前实时天气", "parameters": {{"location": 你要查找的地点}}}}<|tool_end|>"""
        self.tools['googlesearch'] = """<|tool_start|>{{"name": "google_search", "description": "用于使用搜索引擎搜索相关信息", "parameters": {{"searchcontent": 你要搜索的内容}}}}<|tool_end|>"""
        self.tools['resume2webpage'] = """<|tool_start|>{{"name": "resume_to_webpage", "description": "用于将简历转换成个人网页", "parameters": {{}}}}<|tool_end|>"""
        
    def input_tools_prompt(self, tools_select):
        for item in tools_select:
            self.tools_prompt += self.tools[item]
    
    def get_action_messages(self, prompt, content):
        messages = [{'role': 'system', 'content': self.action_prompt.format(context=content)}]
        messages.append({'role': 'user', 'content': prompt})
        return messages
    
    def get_messages(self, prompt_content):
        system_content = self.system_prompt.format(prompt=self.tools_prompt)
        messages = [{'role': 'system', 'content': system_content}]
        messages.append({'role': 'user', 'content': prompt_content})
        return messages

    def actions(self, content):
        if "arxivsearch" in content:
            arxivsearch = ArxivSearch()
            match = re.search(r'{"keyword": "(.*?)"}', content)
            keyword_value = match.group(1)
            res = arxivsearch.reply(keyword_value)
            return res, True
        elif "baidu_map" in content:
            baidumap = BaiduMap()
            match = re.search(r'{"location": "(.*?)", "target": "(.*?)"}', content)
            location_value = match.group(1)
            target_value = match.group(2)
            res = baidumap.reply(location_value, target_value)
            return res, True
        elif "weather_search" in content:
            weathersearch = WeatherSearch()
            match = re.search(r'{"location": "(.*?)"}', content)
            location_value = match.group(1)
            res = weathersearch.reply(location_value)
            return res, True
        elif "google_search" in content:
            googlesearch = GoogleSearch()
            match = re.search(r'{"searchcontent": "(.*?)"}', content)
            searchcontent_value = match.group(1)
            res = googlesearch.reply(searchcontent_value)
            return res, True
        elif "resume_to_webpage" in content:
            resume2website = RESUME2WEBSITE()
            res = resume2website.reply()
            return res, True
        else:
            return "No content", False

if __name__ == "__main__":
    import requests
    user1_content = "北京今天的天气是什么"
    user2_content = "百度大厦附近的酒店有哪些"
    user3_content = "帮我查mindsearch的论文"
    user4_content = "搜一下巴黎奥运会上中国拿了多少个奖牌"
    user5_content = "根据我的简历生成个人主页"

    # res1 = requests.post(f"http://0.0.0.0:8003/streamchat", json=convert_req_data(user1_content), stream=True)
    # res1_text = ''
    # for line in res1.iter_lines():
    #     if line:
    #         res1_text += line.decode('utf-8') + '\n'
    #         # print(line.decode('utf-8'))
    # result1 = actions(res1_text)
    # final_res1 = requests.post(f"http://0.0.0.0:8003/streamchat", json=convert_final_prompt(user1_content, result1), stream=True)
    # for line in final_res1.iter_lines():
    #     if line:
    #         print(line.decode('utf-8'))
    