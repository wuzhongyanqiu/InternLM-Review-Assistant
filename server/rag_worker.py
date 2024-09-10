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

    def gen_questiondb(self, data):
        self.questions_embed_models = {key: FaissRetriever(path, data) for key, path in EMBED_MODEL_PATHS.items()}
        self.questions_bm25 = BM25(data)

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
    
    def reply_questions(self, keyword):
        faiss_results = []
        for key in self.questions_embed_models:
            faiss_results.extend(self.questions_embed_models[key].GetTopK(keyword, 6) )

        bm25_results = self.questions_bm25.GetBM25TopK(keyword, 6)

        deduplicated_results = self.deduplicate_results(faiss_results, bm25_results)
        
        reranked_results = reRank(
            self.rerank_models['rerank1'],  
            6,
            keyword,
            deduplicated_results
        )
        
        return reranked_results

    def deduplicate_results(self, faiss_results, bm25_results):
        # 先对 faiss_results 去重
        seen_ids = set()
        unique_results = []
        for result in faiss_results:
            if isinstance(result[0], Document):
                doc_id = result[0].metadata['id']
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
                doc_id = result.metadata['id']
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result)
                else:
                    print(f"Duplicate doc_id: {doc_id}")
            else:
                print(f"Unexpected type: {type(result)}")

        return unique_results

if __name__ == "__main__":
    getcontent = GetContent()
