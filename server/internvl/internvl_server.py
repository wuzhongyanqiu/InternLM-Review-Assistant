from server.internvl.internvl_worker import get_image_chat
import json
import sqlite3
import re
from typing import List, Union
import fitz  # PyMuPDF
import os
import shutil

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

def extract_question(text: Union[str, List[str]]) -> List[str]:
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        texts = text
    else:
        raise TypeError("text must be a string or a list of string")
    
    interview_questions = []
    pattern = r"<start>(.*?)<end>"

    for item in texts:
        print(item)
        matches = re.findall(pattern, item)
        interview_questions.extend(matches)

    return interview_questions

def upload_images(image_url: Union[str, List[str]]):
    response = get_image_chat(image_url)
    questions = extract_question(response)
    update_db(questions)

def upload_pdf(upload_pdf_path):
    output_folder = os.path.join(current_dir, '../../storage/upload_pdf')
    pdf_path = upload_pdf_path
    try:
        images_path = convert_pdf_to_image(pdf_path, output_folder)
        upload_images(images_path)
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

def update_db(questions: list[str]):
    db_path = os.path.join(current_dir, '../../storage/db_questions.db')
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY,
        question TEXT NOT NULL
    )
    ''')

    for question in questions:
        cursor.execute("INSERT INTO questions (question) VALUES (?)", (question,))

    conn.commit()

    cursor.close()
    conn.close()

def view_db_contents(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM questions")

    questions = cursor.fetchall()

    for question in questions:
        print(question)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    # upload_pdf(os.path.join(current_dir, '../../storage/Charles.pdf'))
    view_db_contents('/root/Mock-Interviewer/storage/db_questions.db')