from dataclasses import dataclass

@dataclass
class ServerConfigs:
    # ASR、TTS音频路径
    ASR_AUDIO_PATH: str = "./storage/asr.wav"
    TTS_AUDIO_PATH: str = "./storage/tts.wav"

    # model路径
    BASE_MODEL_PATH: str = "./models/InternLM-Interview-Assistant"

    # 简历存放路径
    RESUME_PATH: str = "./storage/resume.pdf"

    # embedding model路径
    EMBED_MODEL_PATH: str = "moka-ai/m3e-base"

    # datas目录路径
    DATAS_FOLDER_PATH: str = "./datas"

    # rerank model路径
    RERANK_MODEL_PATH: str = "BAAI/bge-reranker-large"

    # 持久化路径
    PERSIST_PATH: str = "./storage/"
