#!/usr/bin/env python
# coding: utf-8


from langchain.schema import Document
from langchain_community.vectorstores import Chroma,FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os


class FaissRetriever(object):
    # 初始化文档块索引，然后插入faiss库
    def __init__(self, model_path, data, persist_path=None):
        self.embeddings  = HuggingFaceEmbeddings(
                               model_name = model_path,
                               model_kwargs = {"device":"cuda"}
                               # model_kwargs = {"device":"cuda:1"}
                           )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            # 用Document创建文档对象，包含页面内容和元数据
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        # if persist_path and os.path.exists(persist_path):
        #     # 如果指定了持久化路径并且文件存在，则从磁盘加载索引
        #     self.vector_store = FAISS.load_local(persist_path, self.embeddings)
        # else:
        #     # 否则，创建新的索引
        #     self.vector_store = FAISS.from_documents(docs, self.embeddings)
        #     if persist_path:
        #         self.Save_local(persist_path)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        # 删除对象，释放未被引用的缓存内存
        del self.embeddings
        torch.cuda.empty_cache()

    def Save_local(self, persist_path):
        if not os.path.isdir(persist_path):
            os.makedirs(persist_path)
        self.vector_store.save_local(persist_path)

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k):
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store