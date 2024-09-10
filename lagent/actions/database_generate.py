import os
import re
import sqlite3
import fitz  # PyMuPDF
import shutil
from typing import Union, List
import requests
from lagent.actions.base_action import BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode
from openai import OpenAI

class DatabaseGenerator(BaseAction):
    db_dir = '../tmp_dir'
    db_path = '../tmp_dir/db_questions.db'
    client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:8003/v1')
    model_name = client.models.list().data[0].id

    def __init__(self):
        super().__init__()

    @tool_api
    def generate_ques_database(self) -> dict:
        """创建问题数据库。当你需要生成面试题的题库时，可以使用它。

        Returns:
            :class:`dict`: 请求的状态信息
                * success (bool): 数据库生成是否成功
        """
        try:
            self.handle_dir(self.db_dir)
            return {'success': "问题数据库生成成功"}
        except Exception as exc:
            return ActionReturn(
                errmsg=f'DatabaseGenerator exception: {exc}',
                state=ActionStatusCode.ERROR
            )
            
    def handle_dir(self, file_dir):
        for root, dirs, files in os.walk(file_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                # 根据扩展名处理文件
                if file_name.lower().endswith(('.png', '.jpg')):
                    self.upload_images(file_path)
                elif file_name.lower().endswith('.pdf'):
                    self.upload_pdf(file_path)

    def update_db(self, questions: list[str]):
        db_path = os.path.join(os.path.dirname(__file__), self.db_path)

        try:
            # 连接到数据库（如果文件不存在，sqlite3 会自动创建它）
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 创建表（如果表不存在）
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL
            )
            ''')

            # 插入问题
            for question in questions:
                cursor.execute("INSERT INTO questions (question) VALUES (?)", (question,))

            # 提交更改并关闭连接
            conn.commit()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            cursor.close()
            conn.close()

    def extract_question(self, text: Union[str, List[str]]) -> List[str]:
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, list):
            texts = text
        else:
            raise TypeError("text must be a string or a list of string")
        
        interview_questions = []
        pattern = r"<start>(.*?)<end>"

        for item in texts:
            matches = re.findall(pattern, item)
            interview_questions.extend(matches)

        return interview_questions

    def upload_images(self, image_path: Union[str, List[str]]):
        # Assuming you have a function `get_image_chat` that processes the images
        response = self.get_image_chat(image_path)
        questions = self.extract_question(response)
        self.update_db(questions)

    def upload_pdf(self, pdf_path: str):
        output_folder = os.path.join(os.path.dirname(__file__), 'temp_images')
        try:
            images_path = self.convert_pdf_to_image(pdf_path, output_folder)
            self.upload_images(images_path)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
        
    def convert_pdf_to_image(pdf_path, output_folder, image_format='png'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        images_path = []
        doc = fitz.open(pdf_path)
        
        for page_number in range(len(doc)):

            page = doc[page_number]
            
            img = page.get_pixmap(matrix=fitz.Matrix(1, 1), clip=None)
            
            image_path = f"{output_folder}/page_{page_number + 1}.{image_format}"
            images_path.append(image_path)
            img.save(image_path)
        
        doc.close()
        return images_path

    def get_image_chat(image_url: Union[str, List[str]]):
        responses = []
        if isinstance(image_url, str):
            images_url = [image_url]
        elif isinstance(image_url, list):
            images_url = image_url
        else:
            raise TypeError("image_url must be a string or a list of string")

        for item in images_url:
            response = self.client.chat.completions.create(
            model = self.model_name,
            messages = [{
                'role':
                'user',
                'content': [{
                    'type': 'text',
                    'text': '<image>\n将图里的信息提取出几道面试题，每道面试题必须以<|start|>开始，并且以<|end|>结束，如果没有值得提问的面试题，输出空字符串',
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url':
                        item,
                    },
                }],
            }],
            temperature=0.8,
            top_p=0.8)
            responses.append(response.choices[0].message.content)
            print(response.choices[0].message.content)
        return responses