import datetime

from funasr import AutoModel
from funasr.download.name_maps_from_hub import name_maps_ms as NAME_MAPS_MS
from modelscope import snapshot_download
from modelscope.utils.constant import Invoke, ThirdParty

def load_asr_model():
    model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc-c")
    return model

def process_asr(model, audio_path):
    res = model.generate(input=audio_path, batch_size_s=50, hotword="魔搭")

    try:
        res_str = res[0]["text"]
    except Exception as e:
        print("ASR解析失败")
        return ""

    return res_str
