import copy
import hashlib
import json
import os
import base64
import numpy as np
import re
from scipy.io.wavfile import write

import streamlit as st
import requests
import tempfile
from st_audiorec import st_audiorec

from lagent.actions import ActionExecutor, IPythonInterpreter, ArxivSearch
from lagent.actions import MagicMaker
from lagent.actions.mockinterview import MockInterview
from lagent.actions.quicklyqa import QuicklyQA
from lagent.agents.internlm2_agent import INTERPRETER_CN, META_CN, PLUGIN_CN, Internlm2Agent, Internlm2Protocol
from lagent.llms.lmdeploy_wrapper import LMDeployClient
from lagent.llms.meta_template import INTERNLM2_META as META
from lagent.schema import AgentStatusCode

# Your FastAPI ASR endpoint
ASR_API_URL = "http://localhost:8001/asr"

PIC_PATH = '/root/Mock-Interviewer/lagent/server/SadTalker/examples/source_image/flux_interviewer3.png'
VIDEO_PATH = '/root/Mock-Interviewer/lagent/server/SadTalker'

class SessionState:

    def init_state(self):
        """Initialize session state variables."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []

        action_list = [
            MockInterview(),
            QuicklyQA(),
            MagicMaker(),
            ArxivSearch()
        ]
        st.session_state['plugin_map'] = {
            action.name: action
            for action in action_list
        }
        st.session_state['model_map'] = {}
        st.session_state['model_selected'] = None
        st.session_state['plugin_actions'] = set()
        st.session_state['history'] = []
        st.session_state['audio'] = ''
        st.session_state['temperature'] = 0.5
        st.session_state['top_P'] = 0.5
        st.session_state['top_K'] = 20
        st.session_state['absolute_path'] = ''
        st.session_state['user_input'] = ''
        st.session_state['enable_digital_human'] = False


    def clear_state(self):
        """Clear the existing session state."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None
        st.session_state['file'] = set()
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._session_history = []


class StreamlitUI:

    def __init__(self, session_state: SessionState):
        self.init_streamlit()
        self.session_state = session_state

    def init_streamlit(self):
        """Initialize Streamlit's UI settings."""
        st.header(':robot_face: :blue[Lagent] Review Assistant ', divider='rainbow')
        st.sidebar.title('模型控制')
        st.session_state['file'] = set()
        st.session_state['ip'] = None

    def setup_sidebar(self):
        """Setup the sidebar for model and plugin selection."""
        model_name = st.sidebar.text_input('模型名称：', value='internlm2_5-7b-chat')
        meta_prompt = META_CN
        plugin_prompt = PLUGIN_CN
        model_ip = st.sidebar.text_input('模型IP：', value='127.0.0.1:23333')
        if model_name != st.session_state[
                'model_selected'] or st.session_state['ip'] != model_ip:
            st.session_state['ip'] = model_ip
            model = self.init_model(model_name, model_ip)
            self.session_state.clear_state()
            st.session_state['model_selected'] = model_name
            if 'chatbot' in st.session_state:
                del st.session_state['chatbot']
        else:
            model = st.session_state['model_map'][model_name]

        plugin_name = st.sidebar.multiselect(
            '插件选择',
            options=list(st.session_state['plugin_map'].keys()),
            default=[],
        )

        plugin_action = [
            st.session_state['plugin_map'][name] for name in plugin_name
        ]

        if 'chatbot' in st.session_state:
            if len(plugin_action) > 0:
                st.session_state['chatbot']._action_executor = ActionExecutor(
                    actions=plugin_action)
            else:
                st.session_state['chatbot']._action_executor = None

            st.session_state['chatbot']._protocol._meta_template = meta_prompt
            st.session_state['chatbot']._protocol.plugin_prompt = plugin_prompt

        # if st.sidebar.button('清空对话', key='clear'):
        #     self.session_state.clear_state()

        resume_file = st.sidebar.file_uploader('上传简历')
        uploaded_file = st.sidebar.file_uploader('上传文件')
        # 创建一个复选框，用于选择是否启用数字人
        st.session_state['enable_digital_human'] = st.sidebar.checkbox("启用数字人")
        with st.sidebar:
            # 创建三个按钮并放在一行
            col1, col2, col3 = st.columns(3)
            with col1:
                create_db_button = st.button("创建数据")
            with col2:
                clear_db_button = st.button("清除数据")
            with col3:
                clear_dialog_button = st.button("清空对话")
            if clear_dialog_button:
                self.session_state.clear_state()
            if create_db_button:
                create_db()
                st.success("数据库已创建")
            if clear_db_button:
                clear_db()
                st.success("数据库已删除")

        return model_name, model, plugin_action, uploaded_file, model_ip, resume_file

    def init_model(self, model_name, ip=None):
        """Initialize the model based on the input model name."""
        model_url = f'http://{ip}'
        st.session_state['model_map'][model_name] = LMDeployClient(
            model_name=model_name,
            url=model_url,
            meta_template=META,
            max_new_tokens=1024,
            top_p=0.8,
            top_k=100,
            temperature=0,
            repetition_penalty=1.0,
            stop_words=['<|im_end|>'])
        return st.session_state['model_map'][model_name]

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return Internlm2Agent(
            llm=model,
            protocol=Internlm2Protocol(
                tool=dict(
                    begin='{start_token}{name}\n',
                    start_token='<|action_start|>',
                    name_map=dict(
                        plugin='<|plugin|>', interpreter='<|interpreter|>'),
                    belong='assistant',
                    end='<|action_end|>\n',
                ), ),
            max_turn=7)

    def render_user(self, prompt: str):
        with st.chat_message('user'):
            st.markdown(prompt)

    def render_assistant(self, agent_return):
        with st.chat_message('assistant'):
            for action in agent_return.actions:
                if (action) and (action.type != 'FinishAction'):
                    self.render_action(action)
            st.markdown(agent_return.response)

    def render_plugin_args(self, action):
        action_name = action.type
        args = action.args
        import json
        parameter_dict = dict(name=action_name, parameters=args)
        parameter_str = '```json\n' + json.dumps(
            parameter_dict, indent=4, ensure_ascii=False) + '\n```'
        st.markdown(parameter_str)

    def render_action(self, action):
        st.markdown(action.thought)
        if action.type == 'FinishAction':
            pass
        else:
            self.render_plugin_args(action)
        self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if (isinstance(action.result, dict)):
            if 'text' in action.result:
                st.markdown('```\n' + action.result['text'] + '\n```')
            if 'image' in action.result:
                # image_path = action.result['image']
                for image_path in action.result['image']:
                    image_data = open(image_path, 'rb').read()
                    st.image(image_data, caption='Generated Image')
            if 'video' in action.result:
                video_data = action.result['video']
                video_data = open(video_data, 'rb').read()
                st.video(video_data)
            if 'audio' in action.result:
                audio_data = action.result['audio']
                audio_data = open(audio_data, 'rb').read()
                st.audio(audio_data)
        elif isinstance(action.result, list):
            for item in action.result:
                if item['type'] == 'text':
                    st.markdown('```\n' + item['content'] + '\n```')
                elif item['type'] == 'image':
                    image_data = open(item['content'], 'rb').read()
                    st.image(image_data, caption='Generated Image')
                elif item['type'] == 'video':
                    video_data = open(item['content'], 'rb').read()
                    st.video(video_data)
                elif item['type'] == 'audio':
                    audio_data = open(item['content'], 'rb').read()
                    st.audio(audio_data)
        if action.errmsg:
            st.error(action.errmsg)

#################### ASR ##########################
def record_audio(wav_audio_data):
    # Streamlit's audio recorder will be used here.
    audio_bytes = wav_audio_data

    if audio_bytes:
        print(type(audio_bytes))  # 打印类型，检查是否为字节流
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_file.write(audio_bytes)
            audio_file_path = audio_file.name
        
        return audio_file_path
    return None

def transcribe_audio(audio_file_path):
    # Send the audio file to the FastAPI ASR backend
    try:
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"audio_path": audio_file_path})
        response = requests.post(ASR_API_URL, data=payload, headers=headers)
        response.raise_for_status()
        # Delete the temporary audio file after transcription
        os.remove(audio_file_path)
        return response.json().get("result", None)
    except requests.exceptions.RequestException as e:
        st.error(f"Error during ASR request: {e}")
        return None

###################### TTS ############################
def get_tts_api(req_data: dict):
    # 获取 TTS 结果
    res = requests.post(f"http://0.0.0.0:8002/generate_audio", json=req_data).json()
    audio_data_bytes = base64.b64decode(res['audio_data_base64'])
    sample_rate = res['sample_rate']
    audio_data_np = np.frombuffer(audio_data_bytes, dtype=np.float32)

    # 将音频保存到临时文件夹中
    temp_dir = tempfile.gettempdir()  # 获取系统的临时文件夹
    wav_filename = os.path.join(temp_dir, 'generated_audio.wav')
    
    # 将音频保存为 .wav 文件
    write(wav_filename, sample_rate, audio_data_np)

    return audio_data_np, sample_rate, wav_filename  # 返回文件路径

def get_tts_reqdata(text_input, temperature, top_P, top_K, refine_text_flag=True):
    print(text_input)
    req_data = {
        'text': str(text_input),
        'temperature': float(temperature),
        'top_P': float(top_P),
        'top_K': int(top_K),
        'refine_text_flag': bool(refine_text_flag)
    }
    return req_data

########################## digital human #######################
def extract_last_path(output: str) -> str:
    # 使用正则表达式匹配所有.mp4路径
    paths = re.findall(r'[\w./]+\.mp4', output)
    
    # 如果找到路径，返回最后一个
    if paths:
        return os.path.join(VIDEO_PATH, paths[-1])
    else:
        return None

def gen_digital_human(driven_audio: str, source_image: str):
    import requests

    url = 'http://127.0.0.1:8003/inference/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    data = {
        "driven_audio": driven_audio,
        "source_image": source_image
    }

    response = requests.post(url, json=data, headers=headers)

    print(response.status_code)
    print(response.json()['output'])
    return extract_last_path(response.json()['output'])

########################## DB ###########################
def create_db():
    url = "http://0.0.0.0:8004/rag/gen_contentdb"
    requests.post(url)
    url = "http://0.0.0.0:8004/rag/gen_questiondb"
    requests.post(url)

def clear_db():
    url = "http://0.0.0.0:8004/rag/clear_db"
    requests.post(url)

######################### chat and plugin ###################
def chat(model, plugin_action, uploaded_file, resume_file):
    # 显示对话历史
    # for prompt, agent_return in zip(st.session_state['user'],
    #                                 st.session_state['assistant']):
    #     st.session_state['ui'].render_user(prompt)
    #     st.session_state['ui'].render_assistant(agent_return)

    # Text-based input
    # if user_input := st.chat_input(st.session_state['audio']):

    if st.session_state['audio']:
        user_input = st.session_state['audio']
    else:
        user_input = st.session_state['user_input']

    print('user_input')
    print(user_input)

    if user_input:
        with st.container():
            st.session_state['ui'].render_user(user_input)
        st.session_state['user'].append(user_input)

        st.session_state['audio'] = ''
        if (resume_file):
            st.session_state['file'].add(uploaded_file.name)
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type

            postfix = uploaded_file.name.split('.')[-1]
            # prefix = str(uuid.uuid4())
            prefix = hashlib.md5(file_bytes).hexdigest()
            filename = f'{prefix}.{postfix}'
            file_path = os.path.join(root_dir, 'resume')
            file_path = os.path.join(file_path, filename)
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)
            file_size = os.stat(file_path).st_size / 1024 / 1024
            file_size = f'{round(file_size, 2)} MB'
            # st.write(f'File saved at: {file_path}')
            user_input = [
                dict(role='user', content=user_input),
                dict(
                    role='user',
                    content=json.dumps(dict(path=file_path, size=file_size)),
                    name='file')
            ]
        if isinstance(user_input, str):
            user_input = [dict(role='user', content=user_input)]
        st.session_state['last_status'] = AgentStatusCode.SESSION_READY

        # Add file uploader to sidebar
        if (uploaded_file
                and uploaded_file.name not in st.session_state['file']):

            st.session_state['file'].add(uploaded_file.name)
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            if 'image' in file_type:
                st.image(file_bytes, caption='Uploaded Image')
            elif 'video' in file_type:
                st.video(file_bytes, caption='Uploaded Video')
            elif 'audio' in file_type:
                st.audio(file_bytes, caption='Uploaded Audio')
            # Save the file to a temporary location and get the path

            postfix = uploaded_file.name.split('.')[-1]
            # prefix = str(uuid.uuid4())
            prefix = hashlib.md5(file_bytes).hexdigest()
            filename = f'{prefix}.{postfix}'
            file_path = os.path.join(root_dir, 'datas')
            file_path = os.path.join(file_path, filename)
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)

        st.session_state['last_status'] = AgentStatusCode.SESSION_READY

        ##############################打印中间提示############################
        print(st.session_state['session_history'] + user_input)
        ####################################################################
        for agent_return in st.session_state['chatbot'].stream_chat(
                st.session_state['session_history'] + user_input):
            if agent_return.state == AgentStatusCode.PLUGIN_RETURN:
                with st.container():
                    st.session_state['ui'].render_plugin_args(
                        agent_return.actions[-1])
                    st.session_state['ui'].render_action_results(
                        agent_return.actions[-1])
            elif agent_return.state == AgentStatusCode.CODE_RETURN:
                with st.container():
                    st.session_state['ui'].render_action_results(
                        agent_return.actions[-1])
            elif (agent_return.state == AgentStatusCode.STREAM_ING
                or agent_return.state == AgentStatusCode.CODING):
                # st.markdown(agent_return.response)
                # 清除占位符的当前内容，并显示新内容
                with st.container():
                    if agent_return.state != st.session_state['last_status']:
                        st.session_state['temp'] = ''
                        placeholder = st.empty()
                        st.session_state['placeholder'] = placeholder
                    if isinstance(agent_return.response, dict):
                        action = f"\n\n {agent_return.response['name']}: \n\n"
                        action_input = agent_return.response['parameters']
                        if agent_return.response[
                                'name'] == 'IPythonInterpreter':
                            action_input = action_input['command']
                        response = action + action_input
                    else:
                        response = agent_return.response
                    st.session_state['temp'] = response
                    st.session_state['placeholder'].markdown(
                        st.session_state['temp'])
            elif agent_return.state == AgentStatusCode.END:
                st.session_state['session_history'] += (
                    user_input + agent_return.inner_steps)
                agent_return = copy.deepcopy(agent_return)
                agent_return.response = st.session_state['temp']
                st.session_state['assistant'].append(
                    copy.deepcopy(agent_return))
            st.session_state['last_status'] = agent_return.state

        req_data = get_tts_reqdata(agent_return.response, st.session_state['temperature'], st.session_state['top_P'], st.session_state['top_K'], True)
        audio_data_np, sample_rate, wav_filename = get_tts_api(req_data)

        print(wav_filename)

        # 直接播放音频
        if st.session_state['enable_digital_human']:
            st.session_state['absolute_path'] = gen_digital_human(wav_filename, PIC_PATH)
        else:
            st.audio(audio_data_np, sample_rate=sample_rate)

def main():
    # logger = get_logger(__name__)
    # Initialize Streamlit UI and setup sidebar
    if 'ui' not in st.session_state:
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        session_state = SessionState()
        session_state.init_state()
        st.session_state['ui'] = StreamlitUI(session_state)

    else:
        st.set_page_config(
            layout='wide',
            page_title='lagent-web',
            page_icon='./docs/imgs/lagent_icon.png')
        st.header(':robot_face: :blue[Lagent] Review Assistant ', divider='rainbow')
    _, model, plugin_action, uploaded_file, _, resume_file = st.session_state[
        'ui'].setup_sidebar()
    
    # 行内布局和滑动条
    with st.expander("TTS Advanced Settings"):
        st.session_state['temperature'] = st.slider("Audio temperature", min_value=0.00001, max_value=1.0, value=0.5)
        st.session_state['top_P'] = st.slider("top_P", min_value=0.1, max_value=0.9, step=0.05, value=0.5)
        st.session_state['top_K'] = st.slider("top_K", min_value=1, max_value=20, step=1, value=20)

    st.session_state['user_input'] = st.chat_input('')

    col1, col2 = st.columns(2)
    with col1:
        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            audio_file_path = record_audio(wav_audio_data)
            if audio_file_path:
                transcript = transcribe_audio(audio_file_path)
                st.session_state['audio'] = transcript

    # Initialize chatbot if it is not already initialized
    # or if the model has changed
    
    if 'chatbot' not in st.session_state or model != st.session_state[
            'chatbot']._llm:
        st.session_state['chatbot'] = st.session_state[
            'ui'].initialize_chatbot(model, plugin_action)
        st.session_state['session_history'] = []

    if st.session_state['enable_digital_human']:
        with col1:
            chat(model, plugin_action, uploaded_file, resume_file)
    else:
        with col2:
            chat(model, plugin_action, uploaded_file, resume_file)
        
    with col2:
        if st.session_state['absolute_path']:
            st.video(st.session_state['absolute_path'])  # 使用 st.video() 播放 mp4 文件
            st.session_state['absolute_path'] = ''

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(root_dir, 'tmp_dir')
    os.makedirs(root_dir, exist_ok=True)
    main()
