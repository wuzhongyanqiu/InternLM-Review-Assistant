from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from server.tts.tts_worker import generate_audio
import base64

class AudioRequest(BaseModel):
    text: str
    temperature: float
    top_P: float
    top_K: int
    refine_text_flag: bool

app = FastAPI()

@app.post("/generate_audio")
async def get_generate_audio(request: AudioRequest):
    try:
        audio_data_bytes, sample_rate = generate_audio(
            request.text,
            request.temperature,
            request.top_P,
            request.top_K,
            request.refine_text_flag
        )
        audio_data_base64 = base64.b64encode(audio_data_bytes).decode('utf-8')
        return {"audio_data_base64": audio_data_base64, "sample_rate": sample_rate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check")
async def get_check():
    return "tts server 成功开启！"

if __name__ == "__main__":
    import uvicorn
    uvicorn(app, host="0.0.0.0", port=8002)

# uvicorn server.tts.tts_server:app --host 0.0.0.0 --port 8002

