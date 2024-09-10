import sqlite3
from lagent.actions.base_action import BaseAction, tool_api
from lagent.schema import ActionReturn, ActionStatusCode
import os

class SelectQuestion(BaseAction):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    db_path = os.path.join(current_dir, "../../tmp_dir/db_questions.db")

    def __init__(self):
        super().__init__()

    @tool_api
    def generate_question(self) -> dict:
        """从数据库中随机选择一个问题返回，当你需要出一道面试题时可以使用它，你可以根据得到的结果改写。

        Returns:
            :class:`dict`: 包含选定问题的字典
                * question (str): 随机选中的问题
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM questions ORDER BY RANDOM() LIMIT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            if result:
                return {'question': result[1].strip()}
            else:
                return ActionReturn(
                    errmsg='数据库中没有找到问题',
                    state=ActionStatusCode.NO_DATA_FOUND)
        except Exception as exc:
            return ActionReturn(
                errmsg=f'SelectQuestionTool exception: {exc}',
                state=ActionStatusCode.DB_ERROR)

if __name__ == "__main__":
    pass
    

