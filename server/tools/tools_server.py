from fastapi import FastAPI
from pydantic import BaseModel, Field

from server.tools.tools_worker import SelectQuestionTool, AnswerEvaluationTool, ParsingResumesTool

app = FastAPI()
selectquestiontool = SelectQuestionTool()
answerevaluationtool = AnswerEvaluationTool()
parsingresumestool = ParsingResumesTool()

class ToolsItem(BaseModel):
    toolname: str = Field(default='', examples='selectquestiontool')
    query: str = Field(default='', examples='my question is...')
    ans: str = Field(default='', examples='the answer is')
    rag_content: str = Field(default='')
    document_path: str = Field(default='', examples='filepath')

@app.post("/tools")
async def get_tools_res(tools_item: ToolsItem):
    if tools_item.toolname == 'selectquestiontool':
        final_prompt = selectquestiontool.reply()
        return {"result": final_prompt}
    elif tools_item.toolname == 'answerevaluationtool':
        final_prompt = answerevaluationtool.reply(query=tools_item.query, ans=tools_item.ans, rag_content=tools_item.rag_content)
        return {"result": final_prompt}
    elif tools_item.toolname == 'parsingresumestool':
        result = parsingresumestool.reply(document_path=tools_item.document_path).strip()
        return {"result": result}
    else:
        return {"result": "没有这个工具！"}

@app.post("/tools/gen_database")
async def get_tools_res():
    answerevaluationtool.gen_database()

@app.get("/check")
async def get_check():
    return "tools server 成功开启！"


if __name__ == "__main__":
    import uvicorn
    uvicorn(app, host="0.0.0.0", port=8004)

# uvicorn server.tools.tools_server:app --host 0.0.0.0 --port 8004