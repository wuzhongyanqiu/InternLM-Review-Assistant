from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import torch

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class reRankLLM(object):
    def __init__(self, model_path, max_length = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.half()
        self.model.cuda()
        self.max_length = max_length

    def predict(self, query, docs):
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length).to("cuda")
        with torch.no_grad():
            scores = self.model(**inputs).logits
        scores = scores.detach().cpu().clone().numpy()
        response = [doc for score, doc in sorted(zip(scores, docs), reverse=True, key=lambda x:x[0])]
        return response

