import requests
import json
import os
class GoogleSearch:
    def __init__(self):
        self.url = "https://google.serper.dev/search"
        self.headers = {
            'X-API-KEY': os.environ['GOOGLE_SEARCH_KEY'],
            'Content-Type': 'application/json'
        }
        self.description = "谷歌搜索"

    def reply(self, query):
        payload_dict = {"q": query}
        payload = json.dumps(payload_dict)
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
        result = self.convert_reply(json.loads(response.text))
        return result

    def convert_reply(self, text):
        context = ""
        for item in text['organic']:
            context = context + item['snippet'] + "\n"
        return context

if __name__ == "__main__":
    googlesearch = GoogleSearch()
    query = "巴黎奥运会中国拿了多少个奖牌"
    result = googlesearch.reply(query)
    # print(text, "\n")
    print(result)




