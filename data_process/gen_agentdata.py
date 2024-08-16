from base_agent import BaseAgent
import os 
from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
import random
import json
import re
_ = load_dotenv(find_dotenv())
baseagent = BaseAgent()
baseagent.input_tools_prompt(['arxivsearch', 'baidumap', 'weathersearch', 'googlesearch', 'resume2webpage'])
system_prompt = baseagent.system_prompt.format(prompt=baseagent.tools_prompt)

ZHIPUAI_API_KEY = os.environ['ZHIPUAI_API_KEY']
client = ZhipuAI(
    api_key=ZHIPUAI_API_KEY
)
def get_completion_glm(messages, model="glm-4", temperature=0.95):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

conversations = []
q = ['查找<|start|>{}<|end|>的论文', '帮我查<|start|>{}<|end|>附近有什么{}', '帮我搜<|start|>{}|><|end|>', '查询<|start|>{}<|end|>现在的天气', '把我的简历转换成个人主页']
a = ["""{"name": "arxivsearch", "parameters": {"keyword": }}""", """{"name": "baidu_map", "parameters": {"location": , "target": }}""",
"""{"name": "google_search", "parameters": {"searchcontent": }}""","""{"name": "weather_search", "parameters": {"location": }}""",
"""{"name": "resume_to_webpage", "parameters": {}}"""]

from datasets import load_dataset

ds = load_dataset("hfl/cmrc2018")

searchcontents = ds['train']['question'][10000:20000]

import requests
from bs4 import BeautifulSoup

URL = 'https://arxiv.org/list/math.NT/recent'
# https://arxiv.org/list/math.NT/recent
# https://arxiv.org/archive/physics
# https://arxiv.org/list/q-bio/new

response = requests.get(URL)
response.raise_for_status() 

soup = BeautifulSoup(response.text, 'html.parser')

keywords_divs = soup.find_all('div', class_='mathjax')

keywords = []

for div in keywords_divs:
    keywords_text = div.get_text(strip=True)

    keyword_list = keywords_text.split(',')
    keywords.extend([keyword.strip() for keyword in keyword_list])

keywords = list(set(keywords))

pattern = re.compile(r'[ ,.]+')

final_keywords = sum([pattern.split(item) for item in keywords], [])

location = '王府井、鸟巢、前门、五棵松、三里屯、天安门、国贸、西单、中关村、金融街、什刹海、南锣鼓巷、朝阳公园、798艺术区、颐和园、圆明园、香山、八达岭长城、北海公园、雍和宫、大栅栏、天坛、北京动物园、首都国际机场、北京南站、北京西站'
city = '鄂尔多斯、秦皇岛、淄博、烟台、潍坊、临沂、洛阳、南通、襄阳、宜昌、株洲、九江、赣州、中山、惠州、江门、湘潭、株洲、湛江、梅州、茂名、宜昌、襄阳、芜湖、蚌埠、淮南、安庆、泉州、漳州、宁德、景德镇、萍乡、新余、鹰潭、赣州、日照、东营、济宁、泰安、威海、莱芜、临沂、德州、聊城、滨州、菏泽、枣庄、莱芜、赣州、湛江、惠州、江门、肇庆、清远、揭阳、梅州、汕尾、河源、阳江、茂名、韶关、江门、云浮、遵义、六盘水、安顺、铜仁、毕节、黔西南、黔东南、黔南、曲靖、玉溪、保山、昭通、丽江、普洱、临沧、德宏、怒江、迪庆、大理、楚雄、红河、文山、西双版纳、咸阳、铜川、宝鸡、咸阳、渭南、延安、汉中、榆林、安康、商洛、嘉峪关、金昌、白银、天水、武威、张掖、平凉、酒泉、庆阳、定西、陇南、临夏、甘南、海东、海北、黄南、海南、果洛、玉树、海西、石嘴山、吴忠、固原、中卫、克拉玛依、吐鲁番、哈密、昌吉、博尔塔拉、巴音郭楞、阿克苏、克孜勒苏、喀什、和田、伊犁、塔城、阿勒泰、保山、楚雄、德宏、迪庆、红河、丽江、临沧、怒江、普洱、曲靖、文山、西双版纳、玉溪、延安、榆林、张掖、郑州、中山、重庆、驻马店、南阳、开封、周口、商丘、信阳、安阳、平顶山、许昌、新乡、焦作、濮阳、漯河、三门峡、洛阳、济源、南京、无锡、徐州、常州、苏州、南通、连云港、淮安、盐城、扬州、镇江、泰州、宿迁、嘉兴、湖州、绍兴、金华、衢州、舟山、台州、丽水、合肥、芜湖、蚌埠、淮南、马鞍山、淮北、铜陵、安庆、黄山、滁州、六安、宣城、池州、宿州、哈尔滨、大庆、齐齐哈尔、佳木斯、鸡西、鹤岗、双鸭山、牡丹江、伊春、七台河、黑河、绥化、大兴安岭、上海、天津、重庆、石家庄、太原、呼和浩特、沈阳、大连、长春、哈尔滨、南京、无锡、苏州、南通、杭州、宁波、温州、嘉兴、湖州、绍兴、金华、衢州、舟山、台州、丽水、合肥、芜湖、蚌埠、淮南、马鞍山、淮北、铜陵、安庆、黄山、滁州、六安、宣城、池州、宿州、南昌、景德镇、萍乡、九江、新余、鹰潭、赣州、吉安、宜春、抚州、上饶、济南、青岛、淄博、枣庄、东营、烟台、潍坊、济宁、泰安、威海、日照、临沂、德州、聊城、滨州、菏泽、莱芜、潍坊、烟台、济宁、泰安、威海、日照、临沂、德州、聊城、滨州、菏泽、莱芜、郑州、开封、洛阳、平顶山、安阳、鹤壁、新乡、焦作、濮阳、许昌、漯河、三门峡、南阳、商丘、信阳、周口、驻马店、济源、武汉、黄石、十堰、宜昌、襄阳、鄂州、荆门、孝感、荆州、黄冈、咸宁、随州、恩施、仙桃、潜江、天门、神农架、长沙、株洲、湘潭、衡阳、邵阳、岳阳、常德、张家界、益阳、郴州、永州、怀化、娄底、湘西、广州、韶关、深圳、珠海、汕头、佛山、江门、湛江、茂名、肇庆、惠州、梅州、汕尾、河源、阳江、清远、东莞、中山、潮州、揭阳、云浮、成都、自贡、攀枝花、泸州、德阳、绵阳、广元、遂宁、内江、乐山、南充、眉山、宜宾、广安、达州、雅安、巴中、资阳、阿坝、甘孜、凉山、贵阳、六盘水、遵义、安顺、铜仁、毕节、黔西南、黔东南、黔南、昆明、曲靖、玉溪、保山、昭通、丽江、普洱、临沧、楚雄、红河、文山、西双版纳、大理、德宏、怒江、迪庆、拉萨、昌都、山南、日喀则、那曲、阿里、林芝、西安、铜川、宝鸡、咸阳、渭南、延安、汉中、榆林、安康、商洛、兰州、嘉峪关、金昌、白银、天水、武威、张掖、平凉、酒泉、庆阳、定西、陇南、临夏、甘南、西宁、海东、海北、黄南、海南、果洛、玉树、海西、银川、石嘴山、吴忠、固原、中卫、乌鲁木齐、克拉玛依、吐鲁番、哈密、昌吉、博尔塔拉、巴音郭楞、阿克苏、克孜勒苏、喀什、和田、伊犁、塔城、阿勒泰、台北、高雄、基隆、台中、台南、新竹、嘉义、宜兰、桃园、苗栗、彰化、南投、云林、屏东、台东、花莲、澎湖'
target = '电影院、书店、咖啡馆、健身房、艺术画廊、花店、面包房、购物中心、游乐园、博物馆、图书馆、科技馆、水族馆、动物园、植物园、主题公园、溜冰场、攀岩馆、保龄球馆、网咖、美甲店、理发店、按摩店、瑜伽馆、舞蹈工作室、音乐学校、语言中心、旅行社、银行、保险公司、邮局、医院、药店、牙科诊所、宠物店、宠物医院、汽车展厅、自行车店、摩托车店、船只租赁、飞机模型店、天文馆、历史纪念馆、文化中心、社区中心、市政厅、警察局、消防站、法院、大使馆、领事馆、教堂、清真寺、犹太教堂、佛教寺庙、道教观、纪念碑、塔楼、城堡、要塞、城墙、战场遗址、陵墓、墓园、纪念碑、雕塑园、纪念碑谷、历史遗迹、考古遗址、自然公园、国家公园、自然保护区、地质公园、植物园、野生动物保护区、湿地保护区、沙漠保护区、热带雨林、珊瑚礁、海洋公园、河流、湖泊、水库、瀑布、温泉、火山、山脉、山峰、高原、平原、盆地、峡谷、洞穴、岩洞、冰川、极地、热带、亚热带、温带、寒带、气候研究站、海洋研究所、天文台、气象站、地震监测站、生物多样性研究中心、环境保护组织、可持续发展项目、生态旅游区、自然保护区、绿色能源发电站、太阳能农场、风电场、水电站、潮汐能发电站、地热能发电站、核能发电站、石油开采场、天然气田、煤矿、铁矿、铜矿、金矿、钻石矿、宝石矿、稀土矿、盐矿、石墨矿、石灰石矿、大理石矿、花岗岩矿、沙石矿、砾石矿、粘土矿、陶瓷厂、玻璃厂、钢铁厂、铝厂、铜厂、锌厂、铅厂、镍厂、钛厂、镁厂、水泥厂、砖厂、陶瓷厂、造纸厂、印刷厂、塑料制品厂、橡胶制品厂、纤维制品厂、纺织品厂、服装厂、鞋厂、帽厂、手套厂、围巾厂、皮带厂、箱包厂、皮具厂、珠宝首饰厂、手表厂、眼镜厂、化妆品厂、香水厂、个人护理品厂、清洁用品厂、洗涤剂厂、消毒产品厂、医疗器械厂、药品制造厂、生物技术公司、基因编辑实验室、生物制药公司、疫苗研发中心、健康食品厂、营养补充品厂、有机食品市场、素食餐厅、纯素食品店、无麸质食品店、低卡路里食品店、运动营养品店、婴儿食品厂、儿童食品厂、老年食品店、保健食品店、滋补品店、草药店、中药房、针灸诊所、中医医院、康复中心、疗养院、温泉度假村、海滨度假村、山区度假村、乡村度假村、湖畔度假村、河畔度假村、森林度假村、岛屿度假村、沙漠度假村、极地度假村、热带度假村、亚热带度假村、温带度假村、寒带度假村、气候度假村、天文度假村、地质度假村、生态度假村、野生动物观察站、植物观察站、昆虫观察站、鸟类观察站、海洋生物观察站、天文观测站、地质观测站、气象观测站、生物多样性观测站、环境保护观测站、可持续发展观测站、生态旅游观测站'

locations = location.split('、')
citys = city.split('、')
targets = target.split('、')

prompt_a = """你是一个擅长填空的专家，你会按照我给的示例的规则和给定上下文进行填空，你只输出填空后的内容。\n
##规则
给定上下文：查找mindsearch的论文\n
要填空的句子：{"name": "arxivsearch", "parameters": {"keyword": }}\n
根据给定上下文填空后的句子：{"name": "arxivsearch", "parameters": {"keyword": "mindsearch"}}\n

给定上下文：帮我查天安门附近有什么好吃的\n
要填空的句子：{"name": "baidu_map", "parameters": {"location": , "target": }}\n
根据给定上下文填空后的句子：{"name": "baidu_map", "parameters": {"location": "天安门", "target": "美食"}}\n

给定上下文：怎么配置vscode的python环境\n
要填空的句子：{"name": "google_search", "parameters": {"searchcontent": }}\n
根据给定上下文填空后的句子：{"name": "google_search", "parameters": {"searchcontent": "配置vscode的python环境"}}\n

给定上下文：天津现在的天气是什么\n
要填空的句子：{"name": "weather_search", "parameters": {"location": }}\n
根据给定上下文填空后的句子：{"name": "weather_search", "parameters": {"location": "天津"}}\n
# """

for i in range(500):
    for i in range(5):
        conversation = {}
        datas = []
        data = {}
        data['system'] = system_prompt
        if i == 0:
            random_keyword = random.choice(final_keywords)
            messages_q = [{'role': 'system', 'content': 'you are a helpful assistant'}, {'role': 'user', 'content': '原句子\n{}\n, 将句子填空后改写，注意，只输出改写后的完整句子，不要说其他的内容，原句子也不要复述！！！'.format(q[i].format(random_keyword))}]
            convert_q = get_completion_glm(messages_q)
            print(convert_q)
            data['input'] = convert_q
            messages_a = [{'role': 'system', 'content': prompt_a}, {'role': 'user', 'content': '给定上下文：{}\n要填空的句子：{}\n根据给定上下文填空后的句子：'.format(convert_q, a[i])}]
            convert_a = get_completion_glm(messages_a)
            data['output'] = convert_a
            print(convert_a)
        if i == 1:
            random_location = random.choice(locations)
            random_target = random.choice(targets)
            messages_q = [{'role': 'system', 'content': 'you are a helpful assistant'}, {'role': 'user', 'content': '将句子{}改写，注意，只输出改写后的句子，不要说其他的内容，原句子也不要复述，更不要加修饰词！！！'.format(q[i].format(random_location, random_target))}]
            convert_q = get_completion_glm(messages_q)
            print(convert_q)
            data['input'] = convert_q
            messages_a = [{'role': 'system', 'content': prompt_a}, {'role': 'user', 'content': '给定上下文：{}\n要填空的句子：{}\n根据给定上下文填空后的句子：'.format(convert_q, a[i])}]
            convert_a = get_completion_glm(messages_a)
            data['output'] = convert_a
            print(convert_a)
        if i == 2:
            random_searchcontent = random.choice(searchcontents)
            messages_q = [{'role': 'system', 'content': 'you are a helpful assistant'}, {'role': 'user', 'content': '将句子{}改写，注意，只输出改写后的句子，不要说其他的内容，原句子也不要复述，更不要加修饰词！！！'.format(q[i].format(random_searchcontent))}]
            convert_q = get_completion_glm(messages_q)
            print(convert_q)
            data['input'] = convert_q
            messages_a = [{'role': 'system', 'content': prompt_a}, {'role': 'user', 'content': '给定上下文：{}\n要填空的句子：{}\n根据给定上下文填空后的句子：'.format(convert_q, a[i])}]
            convert_a = get_completion_glm(messages_a)
            data['output'] = convert_a
            print(convert_a)
        if i == 3:
            random_city = random.choice(citys)
            messages_q = [{'role': 'system', 'content': 'you are a helpful assistant'}, {'role': 'user', 'content': '将句子{}改写，注意，只输出改写后的句子，不要说其他的内容，原句子也不要复述，更不要加修饰词！！！'.format(q[i].format(random_city))}]
            convert_q = get_completion_glm(messages_q)
            print(convert_q)
            data['input'] = convert_q
            messages_a = [{'role': 'system', 'content': prompt_a}, {'role': 'user', 'content': '给定上下文：{}\n要填空的句子：{}\n根据给定上下文填空后的句子：'.format(convert_q, a[i])}]
            convert_a = get_completion_glm(messages_a)
            data['output'] = convert_a
            print(convert_a)
        if i == 4:
            messages = [{'role': 'system', 'content': 'you are a helpful assistant'}, {'role': 'user', 'content': '将句子{}改写，注意，只输出改写后的句子，不要说其他的内容，原句子也不要复述，更不要加修饰词！！！'.format(q[i])}]
            convert_q = get_completion_glm(messages)
            print(convert_q)
            data['input'] = convert_q
            data['output'] = a[i]
            print(a[i])

        conversation['conversation'] = [data]
        conversations.append(conversation)

# print(conversations)

with open('../datas/agent_data1.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)

# if __name__ == "__main__":
