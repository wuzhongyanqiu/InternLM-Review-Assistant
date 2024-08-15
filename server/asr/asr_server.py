from fastapi import FastAPI
from pydantic import BaseModel

from .asr_worker import load_asr_model, process_asr

app = FastAPI()
model = load_asr_model()

class ASRItem(BaseModel):
    request_id: str  # 请求 ID
    audio_path: str  # wav 文件路径


@app.post("/asr")
async def get_asr(asr_item: ASRItem):
    # 语音转文字
    result = ""
    status = "success"

    result = process_asr(model, asr_item.audio_path)

    return {"request_id": asr_item.request_id, "status": status, "result": result}


# uvicorn server.asr.asr_server:app --host 0.0.0.0 --port 8001
