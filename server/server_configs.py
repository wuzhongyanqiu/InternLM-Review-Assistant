from dataclasses import dataclass

@dataclass
class ServerConfigs:
    # ASR、TTS音频路径
    ASR_AUDIO_PATH: str = "/root/Mock-Interviewer/audio/asr.wav"
    TTS_AUDIO_PATH: str = "/root/Mock-Interviewer/audio/tts.wav"
    # model路径
    BASE_MODEL_PATH: str = "/root/Mock-Interviewer/work_dirs/internlm2_chat_7b_qlora_interview_data/iter_250_merge"
    # 简历存放路径
    RESUME_PATH: str = "/root/Mock-Interviewer/agent/upload_resume.pdf"
    # embedding model路径
    EMBED_MODEL_PATH: str = "moka-ai/m3e-base"
    # datas目录路径
    DATAS_FOLDER_PATH: str = "/root/Mock-Interviewer/datas"
    # rerank model路径
    RERANK_MODEL_PATH: str = "BAAI/bge-reranker-large"
    # 持久化路径
    PERSIST_PATH: str = "/root/Mock-Interviewer/datas_processed/"
