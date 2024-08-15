from fastapi import FastAPI
from pydantic import BaseModel

from .tts_worker import load_tts_model, save_tts_text

app = FastAPI()
model = load_tts_model()

class TTSItem(BaseModel):
    request_id: str  # 请求 ID
    tts_path: str  # wav 文件路径
    tts_text: str  # 要转语音的文本


@app.post("/tts")
async def get_tts(tts_item: TTSItem):
    # 语音转文字
    result = ""
    status = "success"

    result = save_tts_text(tts_item.tts_text, model, tts_item.tts_path)

    return {"request_id": tts_item.request_id, "status": status, "result": result}


# uvicorn server.tts.tts_server:app --host 0.0.0.0 --port 8002