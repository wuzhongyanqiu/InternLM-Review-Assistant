from dataclasses import dataclass
import os
app_script_path = os.path.abspath(__file__)
project_root_path = os.path.dirname(app_script_path)

@dataclass
class Configs:
    # 所有文件的路径
    PROJECT_ROOT = project_root_path

    # ASR、TTS音频路径
    ASR_AUDIO_PATH: str = "./storage/asr.wav"
    TTS_AUDIO_PATH: str = "./storage/tts.wav"

    # TTS model路径
    TTS_MODEL_PATH: str = "./XTTS-v2"
    TTS_CONFIG_PATH: str = "./XTTS-v2/config.json"
    TTS_PERFORM_PATH: str = "./storage/tts_perform.wav"

    # 简历存放路径
    RESUME_PATH: str = "./storage/resume.pdf"

    # 问题 database 路径
    DB_PATH: str = "./storage/db_questions.db"


