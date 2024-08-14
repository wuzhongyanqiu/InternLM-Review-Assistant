from fastapi import FastAPI
from pydantic import BaseModel
from lmdeploy import pipeline
from ..server_configs import ServerConfigs

def load_model():
    model = pipeline(ServerConfigs.BASE_MODEL_PATH)
    return model

app = FastAPI()
model = load_model()

from fastapi.responses import StreamingResponse

async def get_streaming_data(inputs):
    for result in model.stream_infer(inputs):
        yield result.text

class BaseItem(BaseModel):
    inputs : list # 请求

@app.post("/chat")
def get_chat_res(base_item: BaseItem):
    result = model.chat(base_item.inputs)
    return result

@app.post("/streamchat")
async def get_streamchat_res(base_item: BaseItem):
    stream = get_streaming_data(base_item.inputs)

    return StreamingResponse(stream, media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn(app, host="0.0.0.0", port=8003)

# uvicorn server.base.base_server:app --host 0.0.0.0 --port 8003