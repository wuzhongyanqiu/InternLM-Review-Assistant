import sqlite3
from rag.rerank_model import reRankLLM
from rag.faiss_retriever import FaissRetriever
from rag.bm25_retriever import BM25
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from langchain.schema import Document
# from rag.gen_contentdb import GenContentDB
# from rag.gen_questiondb import GenQuestionDB
import os
from typing import Union, List
import pickle
import random

# gencontentdb = GenContentDB()
# genquesiondb = GenQuestionDB()

# embedding model路径
EMBED_MODEL_PATHS = {
    "model1": "moka-ai/m3e-base",
    "model2": "thenlper/gte-large",
    "model3": "BAAI/bge-large-zh-v1.5",
    "model4": "/root/models/bce-embedding-base_v1"
}

# rerank model路径
RERANK_MODEL_PATHS = {
    "rerank1": "BAAI/bge-reranker-large",
    # "rerank2": "/root/models/bce-reranker-base_v1"
}

def reRank(rerank, top_k, query, deduplicated_results):
    max_length = 4000

    rerank_ans = rerank.predict(query, deduplicated_results)
    rerank_ans = rerank_ans[:top_k]

    emb_ans = ""
    for doc in rerank_ans:
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

class GetContent:
    rerank_models = {key: reRankLLM(path) for key, path in RERANK_MODEL_PATHS.items()}

    def gen_contentdb(self, data):
        self.contexts_embed_models = {key: FaissRetriever(path, data) for key, path in EMBED_MODEL_PATHS.items()}
        self.contexts_bm25 = BM25(data)

    def gen_questiondb(self):
        try:
            self.conn = sqlite3.connect('/root/Mock-Interviewer/lagent/server/storage/database')
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")

    def del_db(self):
        if hasattr(self, 'conn'):
            self.conn.close()  # 确保关闭连接
        del self.contexts_embed_models
        del self.contexts_bm25
        del self.conn

    def reply_comments(self, query):
        faiss_results = []
        for key in self.contexts_embed_models:
            faiss_results.extend(self.contexts_embed_models[key].GetTopK(query, 2) )

        bm25_results = self.contexts_bm25.GetBM25TopK(query, 2)

        deduplicated_results = self.deduplicate_results(faiss_results, bm25_results)
        
        reranked_results = reRank(
            self.rerank_models['rerank1'],  
            2,
            query,
            deduplicated_results,
        )
        
        return reranked_results
    
    def select_questions(self):
        query = f"SELECT question FROM questions"
        cursor = self.conn.cursor()
        cursor.execute(query)
        
        results = cursor.fetchall()
        
        # 提取问题内容
        questions = [item[0] for item in results]
        
        # 随机选择两个问题
        return random.sample(questions, 1)

    def reply_questions(self, keyword):
        # 使用 '%' 通配符来进行模糊匹配
        query = f"SELECT question FROM questions WHERE question LIKE ?"
        like_pattern = f"%{keyword}%"
        
        cursor = self.conn.cursor()
        cursor.execute(query, (like_pattern,))
        
        results = cursor.fetchall()
        
        # 提取问题内容
        questions = [item[0] for item in results]
        
        # 如果匹配到的问题数量小于等于2个，直接返回所有问题
        if len(questions) <= 2:
            return questions
        
        # 随机选择两个问题
        return random.sample(questions, 2)

    def deduplicate_results(self, faiss_results, bm25_results):
        # 先对 faiss_results 去重
        seen_ids = set()
        unique_results = []
        for result in faiss_results:
            if isinstance(result[0], Document):
                print(result[0])
                filename = result[0].metadata['file_name']
                pagenum = result[0].metadata['page_num']
                doc_id = filename + '_' + str(pagenum)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result[0])
                else:
                    print(f"Duplicate doc_id: {doc_id}")
            else:
                print(f"Unexpected type: {type(result[0])}")

        # 再对 bm25_results 去重
        for result in bm25_results:
            if isinstance(result, Document):
                print(result)
                filename = result.metadata['file_name']
                pagenum = result.metadata['page_num']
                doc_id = filename + '_' + str(pagenum)
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result)
                else:
                    print(f"Duplicate doc_id: {doc_id}")
            else:
                print(f"Unexpected type: {type(result)}")
        
        for doc in unique_results:
            print(f"File Name: {doc.metadata['file_name']}, "
                f"File Path: {doc.metadata['file_path']}, "
                f"Page Number: {doc.metadata['page_num']}")

        return unique_results

import os
import requests
from typing import List, Optional, Tuple, Union

class GoogleSearch:
    result_key_for_type = {
        'news': 'news',
        'places': 'places',
        'images': 'images',
        'search': 'organic',
    }

    def __init__(self,
                 api_key: Optional[str] = None,
                 timeout: int = 5,
                 search_type: str = 'search'):
        api_key = os.environ.get('SERPER_API_KEY', api_key)
        if api_key is None:
            raise ValueError(
                'Please set Serper API key either in the environment '
                'as SERPER_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key
        self.timeout = timeout
        self.search_type = search_type

    def search(self, query: str, k: int = 2) -> Tuple[int, Union[dict, str]]:
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json',
        }
        params = {
            'q': query,
        }
        try:
            response = requests.post(
                f'https://google.serper.dev/{self.search_type}',
                headers=headers,
                params=params,
                timeout=self.timeout)
        except Exception as e:
            return -1, str(e)
        return response.status_code, response.json()

    def parse_results(self, results: dict, k: int = 2) -> Union[str, List[str]]:
        snippets = []

        if results.get('answerBox'):
            answer_box = results.get('answerBox', {})
            if answer_box.get('answer'):
                return [answer_box.get('answer')]
            elif answer_box.get('snippet'):
                return [answer_box.get('snippet').replace('\n', ' ')]
            elif answer_box.get('snippetHighlighted'):
                return answer_box.get('snippetHighlighted')

        if results.get('knowledgeGraph'):
            kg = results.get('knowledgeGraph', {})
            title = kg.get('title')
            entity_type = kg.get('type')
            if entity_type:
                snippets.append(f'{title}: {entity_type}.')
            description = kg.get('description')
            if description:
                snippets.append(description)
            for attribute, value in kg.get('attributes', {}).items():
                snippets.append(f'{title} {attribute}: {value}.')

        for result in results[self.result_key_for_type[self.search_type]][:k]:
            if 'snippet' in result:
                snippets.append(result['snippet'])
            for attribute, value in result.get('attributes', {}).items():
                snippets.append(f'{attribute}: {value}.')

        if len(snippets) == 0:
            return ['No good Google Search Result was found']
        return snippets

if __name__ == "__main__":
    getcontent = GetContent()
