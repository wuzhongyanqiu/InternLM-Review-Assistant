from fastapi import FastAPI
from pydantic import BaseModel
from lmdeploy import pipeline
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Dict
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
# model路径
BASE_MODEL_PATH = os.path.join(current_dir, '../../models/InternLM-Interview-Assistant')

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    n: int = 1
    max_new_tokens: int = 2048
    top_p: float = 1.0
    top_k: int = 1
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    random_seed: int = None
    stop_words: List[str] = None
    bad_words: List[str] = None
    min_new_tokens: int = None
    skip_special_tokens: bool = True
    logprobs: int = None

@dataclass
class PytorchEngineConfig:
    model_name: str = ''
    tp: int = 1
    session_len: int = None
    max_batch_size: int = 128
    cache_max_entry_count: float = 0.3
    eviction_type: str = 'recompute'
    prefill_interval: int = 16
    block_size: int = 64
    num_cpu_blocks: int = 0
    num_gpu_blocks: int = 0
    adapters: Dict[str, str] = None
    max_prefill_token_num: int = 4096
    thread_safe: bool = False
    download_dir: str = None
    revision: str = None

def load_model():
    pytochengineconfig = PytorchEngineConfig()
    model = pipeline(BASE_MODEL_PATH, model_name="internlm2-chat-7b", backend_config=pytochengineconfig)
    return model

app = FastAPI()
model = load_model()

from fastapi.responses import StreamingResponse

async def get_streaming_data(inputs):
    generation_config = GenerationConfig()
    for result in model.stream_infer(inputs, gen_config=generation_config):
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