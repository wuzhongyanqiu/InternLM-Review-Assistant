#coding=utf-8
import json
import re

def transform_data(data):
    datas = []

    for item in data:
        for i in range(10):
            datas.append(item['conversation'])
    
    # 转换数据格式
    converted_data = [
        {"conversation": [{
            "input": item['input'],
            "output": item['output']}
        ]} for item in datas
    ]
    return converted_data

if __name__ == "__main__":
    input_json_filename = '../datas/interview_data.json'
    output_json_filename = '../datas/Xtuner_interview_data.json'

    # 读取原始JSON文件
    with open(input_json_filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    converted_data = transform_data(data)

    # 写入转换后的JSON文件
    with open(output_json_filename, 'w', encoding='utf-8') as outfile:
        json.dump(converted_data, outfile, ensure_ascii=False, indent=4)

    print(f"转换完成，JSON文件已保存为 '{output_json_filename}'")