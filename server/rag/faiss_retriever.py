#!/usr/bin/env python
# coding: utf-8

from langchain.schema import Document
from langchain_community.vectorstores import Chroma,FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# 持久化路径
# PERSIST_PATH = os.path.join(current_dir, "../../storage/")
PERSIST_PATH = None

class FaissRetriever(object):
    # 初始化文档块索引，然后插入faiss库
    def __init__(self, model_path, data, persist_path=PERSIST_PATH):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda"}
        )
        self.persist_path = persist_path
        docs = []

        # 处理新的 datas 结构
        for idx, document in enumerate(data):
            # document 是 Document 对象
            page_content = document.page_content
            metadata = document.metadata
            docs.append(Document(page_content=page_content, metadata=metadata))

        # 从文档创建FAISS索引
        if persist_path and os.path.exists(persist_path):
            self.vector_store = FAISS.load_local(persist_path, self.embeddings)
        else:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            if persist_path:
                self.Save_local()
        
        del self.embeddings
        torch.cuda.empty_cache()

    def Save_local(self):
        if not os.path.isdir(self.persist_path):
            os.makedirs(self.persist_path)
        self.vector_store.save_local(self.persist_path)

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k):
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store