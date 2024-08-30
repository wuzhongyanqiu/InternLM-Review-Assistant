import os
# script_path = 'deploy.sh'
# try:
#     os.system(f'bash {script_path}')
# except Exception as e:
#     print(f"An error occurred while executing the script: {e}") 

from config import Configs
import sys
sys.path.append(Configs.PROJECT_ROOT)
import streamlit as st
import torch
from torch import nn
import copy
import warnings
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable, List, Optional
from dataclasses import asdict, dataclass
# from audiorecorder import audiorecorder
from web.api import get_asr_api, get_tts_api

from web.api import get_answerevaluation, get_selectquestion, get_parsingresumes, gen_database
from server.tools.tools_prompt import interview_prompt_page2, interview_prompt_system, transquestion_prompt_template
from lmdeploy import pipeline
from agent.base_agent import BaseAgent
from server.internvl.internvl_server import upload_images, upload_pdf 
from server.base.base_server import upload_other
import tempfile
from werkzeug.utils import secure_filename
from os import path
import shutil
import subprocess

app_script_path = os.path.abspath(__file__)
project_root_path = os.path.dirname(app_script_path)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name_or_path = "./models/InternLM-Interview-Assistant"

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 4096
    top_p: float = 0.75
    temperature: float = 0.1
    do_sample: bool = True
    repetition_penalty: float = 1.000

@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break

def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=2048)
        top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config

@st.cache_resource
def load_model():
    model = (AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              trust_remote_code=True)
    return model, tokenizer

def show_chat_messages(model, tokenizer, real_prompt, visualizer=True):
    if visualizer:
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                additional_eos_token_id=92542,
                **asdict(st.session_state.generation_config),
            ):
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        logger.info("##############记录展示型输出############")
        logger.info(cur_response)
        return cur_response
    else:
        for cur_response in generate_interactive(
            model=model,
            tokenizer=tokenizer,
            prompt=real_prompt,
            additional_eos_token_id=92542,
            **asdict(st.session_state.generation_config),
        ):
            pass
        logger.info("##############记录隐藏型输出############")
        logger.info(cur_response)
        return cur_response


system_prompt = "<s><|im_start|>system\n{system}<|im_end|>\n"
user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
assistant_prompt = '<|im_start|>assistant\n{assistant}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'

def convert_prompt(messages):
    total_prompt = ''
    for message in messages[:-1]:
        cur_content = message['content']
        if message['role'] == 'system':
            cur_prompt = system_prompt.format(system=cur_content)
        elif message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'assistant':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=messages[-1]['content'])
    logger.info("##############记录real_prompt############")
    logger.info(total_prompt)
    return total_prompt

def handle_file(file_path):
    if file_path.lower().endswith(('.png', '.jpg')):
        upload_images(file_path)
    elif file_path.lower().endswith('.pdf'):
        upload_pdf(file_path)
    elif file_path.lower().endswith(('.txt', '.html', '.docx', '.md')):
        upload_other(file_path)

def delete_knowledge_base_directory():
    knowledge_base_dir = 'storage/knowledge'
    if os.path.exists(directory_path):  # 检查目录是否存在
        shutil.rmtree(directory_path)  # 删除目录及其所有内容
        st.success("知识库已被清空")
    st.error("知识库不存在或已空")

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

def init_page1():
    if 'page1messages' not in st.session_state:
        st.session_state.page1messages = []
    if 'question_chat' not in st.session_state:
        st.session_state.question_chat = ''
    if 'text_chat' not in st.session_state:
        st.session_state.text_chat = ''
    if 'audio_chat' not in st.session_state:
        st.session_state.audio_chat = ''  

def init_page2():
    if 'page2messages' not in st.session_state:
        st.session_state.page2messages = []
    if 'realpage2messages' not in st.session_state:
        st.session_state.page2messages = []
    if 'ttsconfig' not in st.session_state:
        st.session_state.ttsconfig = {}
    if 'page2selected' not in st.session_state:
        st.session_state.page2selected = []
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = ''
    if 'resume_content' not in st.session_state:
        st.session_state.resume_content = ''
    if 'page2ans' not in st.session_state:
        st.session_state.page2ans = False

def init_page3():
    if 'page3messages' not in st.session_state:
        st.session_state.page3messages = []
    # 记录插件状态
    if 'tools' not in st.session_state:
        st.session_state.tools = []
    if 'action' not in st.session_state:
        st.session_state.action = False
    if 'action_res' not in st.session_state:
        st.session_state.action_res = ''

def init_st_attribute():
    # 记住用户选择的页面
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Interview-Question-Practice"  # 默认选择第一页

def clicked_clear_history():
    if st.session_state:
        for key in list(st.session_state.keys()):
            del st.session_state[key]

def prepare_page1():
    with st.sidebar:
        st.button('开始提问', on_click=clicked_gen_question)
        st.button('清空问题数据库', on_click=clear_question_database)
        st.button('清空知识文件', on_click=delete_knowledge_base_directory)
        st.button('创建 RAG 知识库', on_click=gen_database)
        st.header('创建你的问题数据库')
        uploaded_files = st.file_uploader("上传知识文件", type=["pdf", "jpg", "png", "txt", "docx", "html", "md"], accept_multiple_files=True)

        if uploaded_files:
            st.write("已上传的文件:")
            knowledge_base_dir = 'storage/knowledge'
            os.makedirs(knowledge_base_dir, exist_ok=True)
            for file in uploaded_files:
                st.write(f"{file.name}")
                file_path = os.path.join(knowledge_base_dir, file.name)
                with open(file_path, 'wb') as f:
                    f.write(file.read())
                try:
                    st.write(f"正在处理{file.name}...")
                    handle_file(file_path)
                    st.success("处理完毕")
                except Exception as e:
                    st.error(f"处理 {file.name} 时发生错误: {e}")
            st.session_state.uploaded_files = None
        
def prepare_page2():
    with st.sidebar:
        st.session_state.ttsconfig['temperature'] = st.slider("Audio temperature", min_value=0.00001, max_value=1.0, value=0.5)
        st.session_state.ttsconfig['top_P'] = st.slider("top_P", min_value=0.1, max_value=0.9, step=0.05, value=0.5)
        st.session_state.ttsconfig['top_K'] = st.slider("top_K", min_value=1, max_value=20, step=1, value=20)
        # 创建一个文件上传器，指定类型为 PDF
        st.session_state.uploaded_file = st.file_uploader("上传你的简历", type=["pdf"])

        st.button('准备好了，开始面试！', on_click=clicked_start_interview)

        features = ['ASR', 'TTS', 'Digital humans']
        selected_features = st.multiselect(
            "请选择要开启的功能：",
            features,
            default = []
        )

        st.session_state.page2selected = selected_features

        if 'ASR' in st.session_state.page2selected:
            start_asr()

def prepare_page3():
    with st.sidebar:
        features = ['arxivsearch', 'baidumap', 'weathersearch', 'googlesearch', 'resume2webpage']
        selected_features = st.multiselect(
            "请选择要开启的功能：",
            features,
            default = []
        )
        st.session_state.tools = selected_features        

def prepare_sidebar_config():
    with st.sidebar:
        st.button('Clear history', on_click=clicked_clear_history)
        st.session_state.selected_page = st.radio("请选择页面:", ["Interview-Question-Practice", "Interview-Practice-Exercise", 'Interview-Agent-Butler'])
        
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
            audio_input.export('./storage/asr.wav', format="wav")
            asr_text = get_asr_api('./storage/asr.wav')
            st.session_state.audio_chat = asr_text

def page1():
    st.header("Interview-Question-Practice", divider='rainbow')
    init_page1()
    prepare_page1()
    
    for message in st.session_state.page1messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])  

    if text_chat := st.chat_input('What is up?'):
        st.session_state.text_chat = text_chat
        with st.chat_message('user'):
            st.markdown(text_chat)
        st.session_state.page1messages.append({
            'role': 'user',
            'content': text_chat,  
        })

        if st.session_state.question_chat:
            page1_text_ans()
        else:
            with st.chat_message('assistant'):
                st.markdown("此页面功能为面试题复习，请先点击提问按钮")

def page1_text_ans(): 
    first_messages = [{'role': 'system', 'content': '你擅长用简单的一两句话回答问题'}, {'role': 'user', 'content': st.session_state.question_chat}]     
    rag_content = st.session_state.question_chat + "\n" + show_chat_messages(model, tokenizer, convert_prompt(first_messages), False)
    final_prompt = get_answerevaluation(query=st.session_state.question_chat, ans=st.session_state.text_chat, rag_content=rag_content)
    final_messages = [{'role': 'system', 'content': interview_prompt_system}, {'role': 'user', 'content': final_prompt}]
    real_prompt = convert_prompt(final_messages)
    cur_response = show_chat_messages(model, tokenizer, real_prompt)
    ans_chat = cur_response
    st.session_state.page1messages.append({
        'role': 'assistant',
        'content': cur_response,  
    })
    torch.cuda.empty_cache()

    st.session_state.text_chat = ''
    st.session_state.question_chat = ''

def clicked_gen_question():
    final_prompt = get_selectquestion()
    if final_prompt == "你还未构建问题数据库":
        with st.chat_message('assistant'):
            st.markdown("请先构建问题数据库")
        return

    final_messages = [{'role': 'system', 'content': transquestion_prompt_template}, {'role': 'user', 'content': '给定句子为{}'.format(final_prompt)}]
    
    cur_response = show_chat_messages(model, tokenizer, convert_prompt(final_messages))

    st.session_state.question_chat = cur_response  

    torch.cuda.empty_cache()

    st.session_state.page1messages.append({
        'role': 'assistant',
        'content': st.session_state.question_chat,  # pylint: disable=undefined-loop-variable
    })

def clear_question_database():
    try:
        os.remove(os.path.join(Configs.PROJECT_ROOT, "storage/db_questions.db"))
    except:
        st.write("数据库已清空")

def page2():
    st.header("Interview-Practice-Exercise", divider='rainbow')
    init_page2()
    prepare_mockinterviewspage()

    for message in st.session_state.page2messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])  

    with st.sidebar:
        st.session_state.ttsconfig['temperature'] = st.slider("Audio temperature", min_value=0.00001, max_value=1.0, value=0.5)
        st.session_state.ttsconfig['top_P'] = st.slider("top_P", min_value=0.1, max_value=0.9, step=0.05, value=0.5)
        st.session_state.ttsconfig['top_K'] = st.slider("top_K", min_value=1, max_value=20, step=1, value=20)

        st.session_state.uploaded_file = st.file_uploader("上传你的简历", type=["pdf"])

        st.button('准备好了，开始面试！', on_click=clicked_start_interview)

    if st.session_state.uploaded_file:
        click_uploader()

    if st.session_state.audio_chat:
        assert st.session_state.resume_content, "请先上传您的简历"
        with st.chat_message('user'):
            st.markdown(st.session_state.audio_chat)

        st.session_state.page2messages.append({
            'role': 'user',
            'content': st.session_state.audio_chat,  
        })

        st.session_state.realpage2messages.append({
            'role': 'user',
            'content': st.session_state.audio_chat,  
        })

        st.session_state.audio_chat = ''
        st.session_state.page2ans = True

    if text_chat := st.chat_input('What is up?'):
        assert st.session_state.resume_content, "请先上传您的简历"
        with st.chat_message('user'):
            st.markdown(text_chat)
        st.session_state.page2messages.append({
            'role': 'user',
            'content': text_chat,  
        })
        st.session_state.realpage2messages.append({
            'role': 'user',
            'content': text_chat,  
        })
        st.session_state.page2ans = True

    if st.session_state.page2ans:
        real_prompt = convert_prompt(st.session_state.realpage2messages)

        cur_response = show_chat_messages(model, tokenizer, real_prompt)

        st.session_state.page2messages.append({
            'role': 'assistant',
            'content': cur_response,  
        })

        st.session_state.realpage2messages.append({
            'role': 'assistant',
            'content': cur_response,  
        })

        if "TTS" in st.session_state.page2selected:
            st.session_state.ttsconfig['text'] = cur_response
            audio_data, sample_rate = get_tts_api(st.session_state.ttsconfig)
            st.audio(audio_data, format='audio/wav', sample_rate=sample_rate)
        
        torch.cuda.empty_cache()

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
    st.session_state.page2realmessages.append({
        'role': 'system',
        'content': interview_prompt_page2
    })
    assert st.session_state.resume_content, "你还未上传简历"
    st.session_state.page2messages.append({
        'role': 'user',
        'content': "我准备好了，请开始面试！",  
    })
    st.session_state.page2realmessages.append({
        'role': 'user',
        'content': "面试官您好，我的简历是-\n{}".format(st.session_state.resume_content)
    })
    
    real_prompt = convert_prompt(st.session_state.page2realmessages)

    cur_response = show_chat_messages(model, tokenizer, real_prompt)

    st.session_state.page2messages.append({
        'role': 'assistant',
        'content': cur_response,  # pylint: disable=undefined-loop-variable
    })

    st.session_state.realpage2messages.append({
        'role': 'assistant',
        'content': cur_response,  # pylint: disable=undefined-loop-variable
    })

    torch.cuda.empty_cache()

def page3():
    st.header("Interview-Agent-Butler", divider='rainbow')
    init_page3()
    prepare_agentpage()
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

        cur_response = show_chat_messages(model, tokenizer, real_prompt)

        st.session_state.page3messages.append({
            'role': 'assistant',
            'content': cur_response,  
        })
    
        st.session_state.action_res, st.session_state.action = baseagent.actions(cur_response)

        torch.cuda.empty_cache()

        if st.session_state.action:
            final_messages = baseagent.get_action_messages(prompt=text_chat, content=st.session_state.action_res, assistant_content=st.session_state.page3messages[-1]['content'])
            print(final_messages)
            cur_response = show_chat_messages(model, tokenizer, real_prompt)

            st.session_state.page3messages.append({
                'role': 'assistant',
                'content': cur_response,  
            })

            torch.cuda.empty_cache()

if __name__ == "__main__":
    print('load model begin.')
    model, tokenizer = load_model()
    print('load model end.')

    st.session_state.generation_config = prepare_generation_config()

    init_st_attribute()

    prepare_sidebar_config()

    if st.session_state.selected_page == "Interview-Question-Practice":
        page1()
    elif st.session_state.selected_page == "Interview-Practice-Exercise":
        page2()
    elif st.session_state.selected_page == "Interview-Agent-Butler":
        page3()

    torch.cuda.empty_cache()
    
