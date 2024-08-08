#coding=utf-8
import json

def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def merge_json(file1,file2,file3):
    data1 = read_json(file1)
    data2 = read_json(file2)
    data3 = read_json(file3)
    merged_data = data1
    for i in range(8):
        merged_data = merged_data + data1
    merged_data = merged_data + data2 + data3
    return merged_data

# 示例
merged_data = merge_json("../datas/multi_interview_data.json", "../datas/Xtuner_interview_data.json", "../datas/assistant.json")


with open("../datas/Xtuner_merged_data.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)