#!/usr/bin/env python
# coding: utf-8

from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
import jieba

class BM25(object):

    def __init__(self, documents):
        docs = []
        full_docs = []

        for idx, doc in enumerate(documents):
            # doc 是 Document 对象
            page_content = doc.page_content
            metadata = doc.metadata
            
            # 如果内容太短则跳过
            if len(page_content) < 5:
                continue
            
            # 分词处理
            tokens = " ".join(jieba.cut_for_search(page_content))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            full_docs.append(Document(page_content=page_content, metadata=metadata))

        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()

    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        print("Total documents:", len(self.full_documents))
        
        for line in ans_docs:
            doc_id = line.metadata["id"]
            if doc_id >= len(self.full_documents) or doc_id < 0:
                print(f"Warning: Document ID {doc_id} is out of range.")
            else:
                ans.append(self.full_documents[doc_id])
        return ans


