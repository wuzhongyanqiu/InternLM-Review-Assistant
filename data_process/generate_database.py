import re
import json
import sqlite3

# 打开文件并读取内容
def read_markdown_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

def find_headers(markdown_content):
    # 匹配一级标题
    first_level_headers = re.findall(r'^#\s*(.*?)(?=\n|#|$)', markdown_content, re.MULTILINE)
    # 匹配二级标题
    second_level_headers = re.findall(r'^##\s*(.*?)(?=\n|#|$)', markdown_content, re.MULTILINE)
    return first_level_headers, second_level_headers

def generate_question(first_level_headers, second_level_headers, n=50):
    all_data = []

    for second_header in second_level_headers:
        all_data.append(second_header)

    return all_data

if __name__ == '__main__':
    # 调用函数，传入Markdown文件的路径
    file_path = '../datas/{}.md'
    FILE_NUM = 10
    data = []
    for i in range(FILE_NUM):
        markdown_content = read_markdown_file(file_path.format(i+1))

        # 打印文件内容
        # print(markdown_content)

        first_level_headers, second_level_headers = find_headers(markdown_content)

        # 打印结果
        print("一级标题:")
        for header in first_level_headers:
            print(header)

        print("\n二级标题:")
        for header in second_level_headers:
            print(header)

        data_part = generate_question(first_level_headers, second_level_headers)

        data = data + data_part

    print(data)

    string_list = data

    # 连接到SQLite数据库
    db_path = '../datas/db_questions.db'
    conn = sqlite3.connect(db_path)

    # 创建一个Cursor对象并使用它执行SQL命令
    cursor = conn.cursor()

    # 创建表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY,
        question TEXT NOT NULL
    )
    ''')

    # 插入数据
    for question in string_list:
        cursor.execute("INSERT INTO questions (question) VALUES (?)", (question,))

    # 提交事务
    conn.commit()

    # 关闭Cursor和Connection
    cursor.close()
    conn.close()