from transformers import AutoTokenizer
from typing import Dict
import torch
import numpy as np

from colbert_model import ColBERT


#
# init the model & tokenizer (using the distilbert tokenizer)
#

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # honestly not sure if that is the best way to go, but it works :)
    return tokenizer

def get_model():
    model = ColBERT.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco")
    return model


class ColBERTModel:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("using device:", self.device)
        self.tokenizer = get_tokenizer()
        self.model = get_model().to(self.device)        
        
        # self.get_key=key_format_fn
        # key = self.get_key(i)
        #     doc_detail = {
        #         'doc_id' : 
        #         'vector'
        #     }
    
    def _preprocess_query(self, data):
        query_input = self.tokenizer(data, return_tensors="pt").to(self.device)
        query_input.input_ids += [103] * 8 # [MASK]
        query_input.attention_mask += [1] * 8
        query_input["input_ids"] = torch.LongTensor(query_input.input_ids).unsqueeze(0)
        query_input["attention_mask"] = torch.LongTensor(query_input.attention_mask).unsqueeze(0)
        return query_input

    def _preprocess_document(self, data):
        doc_input = self.tokenizer(data, return_tensors="pt").to(self.device)
        return doc_input
    
    def _compute_representation(self, emb):
        vecs = self.model.forward_representation(emb)
        vecs = vecs.cpu().detach().numpy()
        return vecs
    
    def compute_document_representation(self, data):        
        emb = self._preprocess_document(data)
        doc_vecs = self._compute_representation(emb).squeeze(0)
        return doc_vecs
    
    def compute_query_representation(self, query):
        emb = self._preprocess_query(query)
        query_vecs = self._compute_representation(emb).squeeze(0)
        return query_vecs
    
    def compute_score(self, query, data):
        doc_input = self._preprocess_document(data)        
        query_input = self._preprocess_query(query)
        
        score = self.model.forward(query_input, doc_input)
        
        # doc_vecs = self._compute_representation(doc_input)
        # query_vecs = self._compute_representation(query_input)
        # query_vecs = self.model.forward_representation(query_input)
        # doc_vecs = self.model.forward_representation(doc_input)
        # print(f"query vec {query_vecs.shape}")
        # print(f"doc vec {doc_vecs.shape}")
        
        # score = self.model.forward_aggregation(
        #     query_vecs,
        #     doc_vecs,
        #     query_input["attention_mask"],
        #     doc_input["attention_mask"]
        # )        
        
        return score.item()
    
    def get_tokens(self, data):
        emb = self._preprocess_document(data)
        return self.tokenizer.convert_ids_to_tokens(
            emb.input_ids[0]
        )
        
    def compute_interaction_map(self, query, data):
        doc_input = self._preprocess_document(data)        
        query_input = self._preprocess_query(query)
        
        query_vecs = self.model.forward_representation(query_input)
        doc_vecs = self.model.forward_representation(doc_input)
        
        score = torch.bmm(query_vecs, doc_vecs.transpose(2,1))
        # print(doc_input.input_ids.shape)
        # print(doc_input.input_ids.numpy()[0].shape)
        # print(type(query_input.input_ids))
        doc_tokens = self.tokenizer.convert_ids_to_tokens(doc_input.input_ids.numpy()[0])
        query_tokens = self.tokenizer.convert_ids_to_tokens(query_input.input_ids)
        print(doc_tokens)
        print(query_tokens)
        score = score.squeeze(0).cpu().detach().numpy()
        return score, doc_tokens, query_tokens
    
    def visualize_interaction(self, query, data):  
        
        query_vecs =model.forward_representation(query_input)
        document_vecs =model.forward_representation(passage_input)    

        doc_ids = passage_input.input_ids.numpy()[0]
        query_ids = query_input.input_ids

        f_score = model.forward_aggregation(
            query_vecs,
            document_vecs,
            query_input["attention_mask"],
            passage_input["attention_mask"]
        )
        print(f_score)

        scores = torch.bmm(query_vecs,document_vecs.transpose(2,1))

        (x_shape, y_shape) = scores.squeeze(0).shape
        print(x_shape, y_shape)

        plt.matshow(
          scores.squeeze(0).cpu().detach().numpy()
        )
        plt.xlabel('doc vecs')
        plt.ylabel('query vecs')
        plt.xticks(range(y_shape), doc_tokens )
        plt.xticks(rotation=90)
        plt.yticks(range(x_shape), query_tokens )    
        return query_vecs, document_vecs, plt
