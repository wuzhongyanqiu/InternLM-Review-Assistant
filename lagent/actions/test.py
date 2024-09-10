import os
import sqlite3
current_dir = os.path.dirname(os.path.abspath(__file__))

db_path = os.path.join(current_dir, "../../tmp_dir/db_questions.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT * FROM questions ORDER BY RANDOM() LIMIT 1")
result = cursor.fetchone()
cursor.close()
conn.close()
if result:
    print(result)