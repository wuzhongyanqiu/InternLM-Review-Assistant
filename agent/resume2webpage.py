import sys
sys.path.append('/root/Mock-Interviewer/') 
import requests
from server.server_configs import ServerConfigs
from web.api import get_parsingresumes

class RESUME2WEBSITE:
    def __init__(self):
        self.system_prompt = """
        - Role: 自动化网页生成器
        - Background: 用户希望通过一次性提供简历描述，自动获得个人主页的HTML和CSS代码。
        - Profile: 你是一个高级自动化工具，能够理解用户描述的简历内容，并将其转化为网页格式。
        - Skills: 能够解析文本描述，将其结构化为HTML和CSS代码，无需人工干预。
        - Goals: 根据用户一次性提供的简历描述，生成一个完整的个人主页，包含所有必要的信息。
        - Constrains: 不需要与用户进行交互，直接根据描述生成代码；不使用JavaScript或其他动态效果。
        - OutputFormat: 生成的代码应包括HTML结构和CSS样式。
        - Workflow:
        1. 解析用户提供的简历描述，提取关键信息。
        2. 根据提取的信息，构建HTML页面结构。
        3. 应用CSS样式，确保页面美观、易读。
        4. 确保所有简历内容被适当地展示在个人主页上。
        - Examples:
        用户可能会提供如下描述：
        - "我叫张三，是一名软件工程师。我在XX大学获得了计算机科学学位，并在YY公司担任过前端开发职位。我精通JavaScript、HTML和CSS。"
        - Initialization: 请输入您的简历描述，我将为您生成个人主页的HTML和CSS代码。
        """
        self.resume = get_parsingresumes(ServerConfigs.RESUME_PATH)

    def reply(self):
        total_prompt = []
        total_prompt.append({'role': 'system', 'content': self.system_prompt})
        total_prompt.append({'role': 'user', 'content': "帮我把我的简历变成html和css的个人主页"})
        total_prompt.append({'role': 'assistant', 'content': "请输入您的简历描述，我将为您生成个人主页的HTML和CSS代码。"})
        total_prompt.append({'role': 'user', 'content': "我的简历内容是：\n{}\n".format(self.resume)})
        req_data = {
            "inputs": total_prompt,
        }
        
        res = requests.post(f"http://0.0.0.0:8003/streamchat", json=req_data, stream=True)
        final_res = ''
        for line in res.iter_lines():
            if line:
                final_res += line.decode('utf-8')
        return final_res

if __name__ == "__main__":
    resume2website = RESUME2WEBSITE()
    result = resume2website.reply()
    print(result)