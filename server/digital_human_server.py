from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
import uvicorn

app = FastAPI()

# 定义数据模型
class InferenceRequest(BaseModel):
    driven_audio: str
    source_image: str

@app.post("/inference/")
async def inference(request: InferenceRequest):
    """
    传入音频和图片路径，调用 inference.py 进行推理。
    :param driven_audio: 音频文件的路径
    :param source_image: 图片文件的路径
    :return: 推理的结果
    """

    # 将工作目录切换到 SadTalker 目录
    sad_talker_dir = "SadTalker"
    os.chdir(sad_talker_dir)

    # 构建执行推理的命令
    command = [
        "python", "inference.py",
        "--driven_audio", request.driven_audio,
        "--source_image", request.source_image,
        "--enhancer", "gfpgan"  # 可根据需要修改
    ]

    # 调用推理脚本并捕获输出
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return {"message": "Inference completed successfully", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"error": "Inference failed", "output": e.stderr}
    finally:
        # 恢复原来的工作目录（可选）
        os.chdir("..")

# 运行 FastAPI 服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)