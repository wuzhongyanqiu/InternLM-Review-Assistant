from config import Configs
import sys
sys.path.append(Configs.PROJECT_ROOT)
import streamlit as st
import torch
from torch import nn
import copy
import warnings
from transformers.utils import logging
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from typing import Callable, List, Optional
from dataclasses import asdict, dataclass
from audiorecorder import audiorecorder
from web.api import get_asr_api, get_tts_api, get_chat_responses, get_streamchat_responses

from web.api import get_answerevaluation, get_selectquestion, get_parsingresumes
from server.tools.tools_prompt import multiinterview_prompt_template
from lmdeploy import pipeline
from agent.base_agent import BaseAgent
from server.tools.tools_prompt import interview_prompt_template_norag
from server.internvl.internvl_server import upload_images, upload_pdf
import tempfile
from werkzeug.utils import secure_filename
from os import path
import shutil

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

app_script_path = os.path.abspath(__file__)
project_root_path = os.path.dirname(app_script_path)

logger = logging.get_logger(__name__)

def handle_image(file_path):
    upload_images(file_path)

def handle_pdf(file_path):
    upload_pdf(file_path)

def combine_multiinterview_history(prompt):
    resumes = st.session_state.resume_content
    total_prompt = []
    total_prompt.append({'role': 'system', 'content': multiinterview_prompt_template.format(resumes)})
    total_prompt = total_prompt + st.session_state.page2messages
    total_prompt.append({'role': 'user', 'content': prompt})
    return total_prompt

def combine_multiinterview():
    resumes = st.session_state.resume_content
    total_prompt = []
    total_prompt.append({'role': 'system', 'content': multiinterview_prompt_template})
    total_prompt.append({'role': 'user', 'content': "您好，面试官，我的简历内容是：\n{}\n".format(resumes)})
    return total_prompt

def init_page3():
    if 'page3messages' not in st.session_state:
        st.session_state.page3messages = []
    # 记录插件状态
    if 'tools' not in st.session_state:
        st.session_state.tools = []
    if 'arxivsearch' not in st.session_state:
        st.session_state.arxivsearch = False
    if 'baidumap' not in st.session_state:
        st.session_state.baidumap = False
    if 'weathersearch' not in st.session_state:
        st.session_state.weathersearch = False
    if 'googlesearch' not in st.session_state:
        st.session_state.googlesearch = False
    if 'resume2webpage' not in st.session_state:
        st.session_state.resume2webpage = False
    if 'action' not in st.session_state:
        st.session_state.action = False
    if 'action_res' not in st.session_state:
        st.session_state.action_res = ''

def init_st_attribute():
    if 'page1messages' not in st.session_state:
        st.session_state.page1messages = []
    if 'page2messages' not in st.session_state:
        st.session_state.page2messages = []
    if 'page3messages' not in st.session_state:
        st.session_state.page3messages = []
    # 记录生成的问题
    if 'question_chat' not in st.session_state:
        st.session_state.question_chat = ''
    # 记住用户直接输入的文本
    if 'user_chat' not in st.session_state:
        st.session_state.user_chat = ''
    # 记住用户选择的页面
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Interview-Assistant-QuestionReview"  # 默认选择第一页
    # 记住用户语音输入的文本
    if 'audio_chat' not in st.session_state:
        st.session_state.audio_chat = ''  
    # 决定是否开启ASR、TTS、digitalhuman
    if 'selected_asr' not in st.session_state:
        st.session_state.selected_asr = False 
    if 'selected_tts' not in st.session_state:
        st.session_state.selected_tts = False  # 默认不开启
    if "selected_digitalhuman" not in st.session_state:
        st.session_state.selected_digitalhuman = False
    # page2 记录是否上传文件、解析后的简历
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = ''
    if 'resume_content' not in st.session_state:
        st.session_state.resume_content = ''

def clicked_clear_history():
    if st.session_state:
        for key in list(st.session_state.keys()):
            del st.session_state[key]

# def prepare_config():
#     with st.sidebar:
#         max_length = st.slider('Max Length',
#                                min_value=8,
#                                max_value=32768,
#                                value=2048)
#         top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
#         temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)
#         st.button('Clear Chat History', on_click=on_btn_click)

#     generation_config = GenerationConfig(max_length=max_length,
#                                          top_p=top_p,
#                                          temperature=temperature)

#     return generation_config

def prepare_questionreviewpage():
    pass
        

def prepare_mockinterviewspage():
    with st.sidebar:
        features = ['ASR', 'TTS', 'Digital humans']
        selected_features = st.multiselect(
            "请选择要开启的功能：",
            features,
            default = []
        )

        if 'ASR' in selected_features:
            st.session_state.selected_asr = True
        else:
            st.session_state.selected_asr = False

        if 'TTS' in selected_features:
            st.session_state.selected_tts = True
        else:
            st.session_state.selected_tts = False

        if 'Digital humans' in selected_features:
            st.session_state.selected_digitalhuman = True
        else:
            st.session_state.selected_digitalhuman = False

        if st.session_state.selected_asr:
            start_asr()

def prepare_agentpage():
    with st.sidebar:
        features = ['arxivsearch', 'baidumap', 'weathersearch', 'googlesearch', 'resume2webpage']
        selected_features = st.multiselect(
            "请选择要开启的功能：",
            features,
            default = []
        )

        if 'arxivsearch' in selected_features:
            st.session_state.arxivsearch = True
            st.session_state.tools.append('arxivsearch')
        else:
            st.session_state.arxivsearch = False

        if 'baidumap' in selected_features:
            st.session_state.baidumap = True
            st.session_state.tools.append('baidumap')
        else:
            st.session_state.baidumap = False

        if 'weathersearch' in selected_features:
            st.session_state.weathersearch = True
            st.session_state.tools.append('weathersearch')
        else:
            st.session_state.weathersearch = False
        
        if 'googlesearch' in selected_features:
            st.session_state.googlesearch = True
            st.session_state.tools.append('googlesearch')
        else:
            st.session_state.googlesearch = False

        if 'resume2webpage' in selected_features:
            st.session_state.resume2webpage = True
            st.session_state.tools.append('resume2webpage')
        else:
            st.session_state.resume2webpage = False
        

def prepare_sidebar_config():
    from PIL import Image
    image = Image.open('image.png')
    with st.sidebar:
        st.image(image, width=250)
        # Radio 按钮选择页面
        st.button('Clear history', on_click=clicked_clear_history)
        st.session_state.selected_page = st.radio("请选择页面:", ["Interview-Assistant-QuestionReview", "Interview-Assistant-MockInterviews", 'Interview-Assistant-Agent'])
        

def start_asr():
    with st.sidebar:
        st.title("Answer for voice")
        audiorecorder_options = {
            "start_prompt": "start",
            "pause_prompt": "pause",
            "stop_prompt": "stop",
            "show_visualizer": True,
        }

        audio_input = audiorecorder(**audiorecorder_options)

        if len(audio_input) > 0:
            audio_input.export(ServerConfigs.ASR_AUDIO_PATH, format="wav")
            asr_text = get_asr_api(ServerConfigs.ASR_AUDIO_PATH)
            st.session_state.audio_chat = asr_text

# 定义页面内容
def page1():
    prepare_questionreviewpage()
    st.header("Interview-Assistant-QuestionReview", divider='rainbow')
    # 显示聊天历史中的所有信息，对于每条信息使用st.chat_message()创建聊天气泡，并用st.markdown()来渲染消息内容
    # print("page1 is load!")
    # print(st.session_state.page1messages)
    for message in st.session_state.page1messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])  

    if text_chat := st.chat_input('What is up?'):
        st.session_state.user_chat = text_chat
        with st.chat_message('user'):
            st.markdown(text_chat)
        st.session_state.page1messages.append({
            'role': 'user',
            'content': text_chat,  
        })

    placeholder = st.empty()
    with st.sidebar:
        st.button('生成问题', on_click=clicked_gen_question)
        st.button('清空问题数据库', on_click=clear_question_database)

        st.header('创建你的问题数据库')
        uploaded_file = st.file_uploader("上传知识文件", type=["pdf", "jpg", "png"])

        if uploaded_file is not None:
            filename = secure_filename(uploaded_file.name)

            temp_dir = tempfile.mkdtemp()

            file_path = path.join(temp_dir, filename)

            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            try:
                if filename.lower().endswith(('.png', '.jpg')):

                    placeholder.markdown("正在处理你的图片...")
                    handle_image(file_path)
                    placeholder.markdown("问题数据库已更新")
                    st.session_state.uploaded_file = None
                elif filename.lower().endswith('.pdf'):
                    placeholder.markdown("正在处理你的文件...")
                    handle_pdf(file_path)
                    placeholder.markdown("问题数据库已更新")
                    st.session_state.uploaded_file = None
            finally:
                shutil.rmtree(temp_dir)

        file_urls = st.text_area("批量输入你的文件url, 每行一个")

        if st.button('处理批量图片'):
            if file_urls:
                urls = file_urls.splitlines()
                handle_image(urls)
        
    if st.session_state.question_chat:
        if st.session_state.audio_chat:
            gen_audio_ans()
        if st.session_state.user_chat:
            gen_text_ans()
    if not st.session_state.question_chat and st.session_state.user_chat:
        gen_normal_ans()

def gen_normal_ans():
    final_messages = combine_history(st.session_state.user_chat)
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        # 调用生成器generate_interactive生成机器人的回复，每产生文本立即在聊天界面显示
        for cur_response in get_streamchat_responses(final_messages):
            # 显示机器人响应
            message_placeholder.markdown(cur_response + '▌')
        message_placeholder.markdown(cur_response)
    st.session_state.page1messages.append({
        'role': 'assistant',
        'content': cur_response,  # pylint: disable=undefined-loop-variable
    })


def gen_text_ans():      
    first_prompt = interview_prompt_template_norag.format(st.session_state.question_chat, st.session_state.user_chat)
    first_messages = [{
        'role': 'user',
        'content': first_prompt
    }]
    rag_content = ''
    for cur_response in get_streamchat_responses(first_messages):
        rag_content += cur_response
    
    print(rag_content)
    final_prompt = get_answerevaluation(query=st.session_state.question_chat, ans=st.session_state.user_chat, rag_content=rag_content)
    print(final_prompt)
    final_messages = [{
        'role': 'user',
        'content': final_prompt
    }]
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        # 调用生成器generate_interactive生成机器人的回复，每产生文本立即在聊天界面显示
        all_response = ''
        for cur_response in get_streamchat_responses(final_messages):
            all_response += cur_response
            # 显示机器人响应
            message_placeholder.markdown(all_response + '▌')
        message_placeholder.markdown(all_response)
    # st.session_state.user_chat = ''
    # st.session_state.question_chat = ''


def gen_audio_ans():
    with st.chat_message('user'):
        st.markdown(st.session_state.audio_chat)

    final_prompt = get_answerevaluation(query=st.session_state.question_chat, ans=st.session_state.audio_chat)
    final_messages = [{
        'role': 'user',
        'content': final_prompt
    }]
    with st.chat_message('assistant'):
        all_response = ''
        message_placeholder = st.empty()
        for cur_response in get_streamchat_responses(final_messages):
            all_response += cur_response
            message_placeholder.markdown(all_response + '▌')
        message_placeholder.markdown(all_response)
    ans_chat = all_response
    # st.session_state.question_chat = ''
    # st.session_state.audio_chat = ''

    if st.session_state.selected_tts:
        _ = get_tts_api(ans_chat, ServerConfigs.TTS_AUDIO_PATH)
        st.audio(ServerConfigs.TTS_AUDIO_PATH)
   

def clicked_gen_question():
    final_prompt = get_selectquestion()
    final_messages = [{
        'role': 'user',
        'content': final_prompt
    }]
    for cur_response in get_streamchat_responses(final_messages):
        pass
    st.session_state.question_chat = cur_response
    st.session_state.page1messages.append({
        'role': 'assistant',
        'content': cur_response,  # pylint: disable=undefined-loop-variable
    })

def clear_question_database():
    os.remove(os.path.join(Configs.PROJECT_ROOT, "storage/db_questions.db"))

def page2():
    prepare_mockinterviewspage()
    st.header("Interview-Assistant-MockInterviews", divider='rainbow')

    for message in st.session_state.page2messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])  

    with st.sidebar:
        # 创建一个文件上传器，指定类型为 PDF
        st.session_state.uploaded_file = st.file_uploader("上传你的简历", type=["pdf"])

        st.button('准备好了，开始面试！', on_click=clicked_start_interview)

    if st.session_state.uploaded_file:
        click_uploader()

    if st.session_state.audio_chat:
        gen_audio_ans()

    if text_chat := st.chat_input('What is up?'):
        assert st.session_state.resume_content, "请先上传您的简历"

        final_messages = combine_multiinterview_history(text_chat)
        with st.chat_message('user'):
            st.markdown(text_chat)
        st.session_state.page2messages.append({
            'role': 'user',
            'content': text_chat,  
        })

        with st.chat_message('assistant'):
            all_response = ''
            message_placeholder = st.empty()
            for cur_response in get_streamchat_responses(final_messages):
                all_response += cur_response
                message_placeholder.markdown(all_response + '▌')
            message_placeholder.markdown(all_response)
        ans_chat = all_response

        st.session_state.page2messages.append({
            'role': 'assistant',
            'content': all_response,  
        })

        if st.session_state.selected_tts:
            _ = get_tts_api(ans_chat, ServerConfigs.TTS_AUDIO_PATH)
            st.audio(ServerConfigs.TTS_AUDIO_PATH)

def click_uploader():
    empty_container = st.empty()
    empty_container.markdown("简历解析中. ..............................")
    with open(Configs.RESUME_PATH, 'wb') as f:
        f.write(st.session_state.uploaded_file.read())
    st.session_state.resume_content = get_parsingresumes(Configs.RESUME_PATH)
    del st.session_state.uploaded_file
    empty_container.markdown("简历解析完成...............................")
    empty_container.empty()  

def clicked_start_interview():
    assert st.session_state.resume_content, "你还未上传简历"
    st.session_state.page2messages.append({
        'role': 'user',
        'content': "我准备好了，请开始面试！",  
    })
    final_messages = combine_multiinterview()

    with st.chat_message('assistant'):
        # message_placeholder = st.empty()
        # 调用生成器generate_interactive生成机器人的回复，每产生文本立即在聊天界面显示
        print(final_messages)
        for cur_response in get_streamchat_responses(final_messages):
            pass
            # 显示机器人响应
        #     message_placeholder.markdown(cur_response + '▌')
        # message_placeholder.markdown(cur_response)
    st.session_state.page2messages.append({
        'role': 'assistant',
        'content': cur_response,  # pylint: disable=undefined-loop-variable
    })

def page3():
    init_page3()
    prepare_agentpage()
    st.header("Interview-Assistant-Agent", divider='rainbow')
    baseagent = BaseAgent()
    baseagent.input_tools_prompt(st.session_state.tools)

    for message in st.session_state.page3messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])  

    with st.sidebar:
        # 创建一个文件上传器，指定类型为 PDF
        st.session_state.uploaded_file = st.file_uploader("上传你的简历", type=["pdf"])

    if text_chat := st.chat_input('What is up?'):
        final_messages = baseagent.get_messages(prompt_content=text_chat)
        print(final_messages)
        with st.chat_message('user'):
            st.markdown(text_chat)
        st.session_state.page3messages.append({
            'role': 'user',
            'content': text_chat,  
        })

        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            all_response = ''
            for cur_response in get_streamchat_responses(final_messages):
                all_response += cur_response
                message_placeholder.markdown(all_response + '▌')
            message_placeholder.markdown(all_response)

            st.session_state.page3messages.append({
                'role': 'assistant',
                'content': all_response,  
            })

            print(all_response)
        
            st.session_state.action_res, st.session_state.action = baseagent.actions(all_response)

        if st.session_state.action:
            final_messages = baseagent.get_action_messages(prompt=text_chat, content=st.session_state.action_res, assistant_content=st.session_state.page3messages[-1]['content'])
            print(final_messages)
            with st.chat_message('assistant'):
                message_placeholder = st.empty()
                all_response = ''
                for cur_response in get_streamchat_responses(final_messages):
                    all_response += cur_response
                    message_placeholder.markdown(all_response + '▌')
                message_placeholder.markdown(all_response)

                st.session_state.page3messages.append({
                    'role': 'assistant',
                    'content': all_response,  
                })

if __name__ == "__main__":

    init_st_attribute()

    prepare_sidebar_config()

    if st.session_state.selected_page == "Interview-Assistant-QuestionReview":
        page1()
    elif st.session_state.selected_page == "Interview-Assistant-MockInterviews":
        page2()
    elif st.session_state.selected_page == "Interview-Assistant-Agent":
        page3()

    torch.cuda.empty_cache()
    
