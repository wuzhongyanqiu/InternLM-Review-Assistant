import requests
import os
class BaiduMap:
    def __init__(self):
        self.base_url = "https://api.map.baidu.com/place/v2/search"
        self.map_url = "https://api.map.baidu.com/staticimage/v2"
        self.ak = os.environ['BAIDU_MAP_KEY']
        self.description = "查找地点附近的内容"
    
    def reply_company(self, query):
        params = {
            "query": query,
            "tag": "公司",
            "region": "北京",
            "output": "json",
            "page_size": 1,
            "ak": self.ak
        }
        response = requests.get(url=self.base_url, params=params)
        return response.json()

    def reply_circle(self, query, location):
        params = {
            "query": query,
            "location": location,
            "radius": "2000",
            "output": "json",
            "ak": self.ak
        }
        response = requests.get(url=self.base_url, params=params)
        return response.json()

    def reply_staticimage(self, center, markers):
        params = {
            # "center": "116.403874,39.914889",
            "center": center,
            "width": "512",
            "height": "256",
            "zoom": "11",
            "markers": markers,
            # "markers": "116.288891,40.004261|116.487812,40.017524|116.525756,39.967111|116.536105,39.872374|116.442968,39.797022|116.270494,39.851993|116.275093,39.935251|116.383177,39.923743",
            "markerStyles": "l,A|m,B|l,C|l,D|m,E|,|l,G|m,H",
            "ak": self.ak
        }
        response = requests.get(url=self.map_url, params=params)
        return response.json()
    
    def reply(self, location, target):
        result1 = self.reply_company(location)
        coordinate = ','.join([str(result1['results'][0]['location']['lat']), str(result1['results'][0]['location']['lng'])])
        result2 = self.reply_circle(target, coordinate)
        # result3 = []
        # for item in result2['results']:
        #     result3.append(','.join([str(item['location']['lng']), str(item['location']['lat'])]))
        # res = '|'.join(result3)
        return result2


if __name__ == "__main__":
    baidumap = BaiduMap()
    query1 = "百度"
    result1 = baidumap.reply_company(query1)
    # print(baidumap.reply_company(query1))
    query2 = "酒店"
    locations = []
    locations_swap = []
    name = []
    for item in result1['results']:
        name.append(item['name'])
        locations.append(','.join([str(item['location']['lng']), str(item['location']['lat'])]))
        locations_swap.append(','.join([str(item['location']['lat']), str(item['location']['lng'])]))

    result2s = []
    for location in locations_swap:
        result2 = []
        result = baidumap.reply_circle(query2, location)
        for item in result['results']:
            result2.append(','.join([str(item['location']['lng']), str(item['location']['lat'])]))
        res = '|'.join(result2)
        result2s.append(res)
    print(result2s)
    final_res = []
    for i in range(len(result2s)):
        print(locations[i])
        print(result2s[i])
        final_res.append(baidumap.reply_staticimage(locations[i], result2s[i]))
    print(final_res)



    
        




