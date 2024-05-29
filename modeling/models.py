import torch
from transformers import DistilBertModel, BertModel, BertConfig
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sentence_transformers


class CustomSTEmbedding(nn.Module):
    """
    Wrapper to obtain bert embeddings from ST models.
    """

    def __init__(self, bert):
        super(CustomSTEmbedding, self).__init__()
        if isinstance(bert, str):
            self.bert = sentence_transformers.SentenceTransformer(bert)
        else:
            self.bert = bert

    def forward(self, input_ids, attention_mask):
        return self.bert({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })['sentence_embedding']


class CustomHFEmbedding(nn.Module):
    """
    Wrapper to obtain bert embeddings from HF models.
    """
    def __init__(self, bert):
        super(CustomHFEmbedding, self).__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):
        _token_embs = self.bert(input_ids, attention_mask)["last_hidden_state"]  # (bs, seq_len, dim)
        _division = torch.sum(attention_mask, dim=1)
        embs = torch.sum(_token_embs * attention_mask.unsqueeze(2), dim=1) / _division.unsqueeze(1)

        return embs

class CustomEmbedding(torch.nn.Module):
    def __init__(self, encoder_type, encoder_name, transform_dim):
        super(CustomEmbedding, self).__init__()
        if(encoder_type == "hf"):
            digits, nondigits = self.split_hf_config(encoder_name)
            assert nondigits == ["L", "H", "D"] and len(digits) == 3, "hf config should be in format xLyHzD for x layers, y attention heads and z hidden dimension"
            digits = [int(x) for x in digits]
            configuration = BertConfig(vocab_size=30522, num_hidden_layers=digits[0], num_attention_heads=digits[1], hidden_size=digits[2], intermediate_size=digits[2] * 2)
            print(f"Using a hf encoder {encoder_name} with config {configuration}")
            self.encoder = CustomHFEmbedding(BertModel(configuration))
        elif(encoder_type == "st"):
            print(f"Using a st encoder {encoder_name}")
            self.encoder = CustomSTEmbedding(encoder_name)
        else:
            raise NotImplementedError("Invalid encoder type")

        self.transform_dim = transform_dim
        if(self.transform_dim > 0):
            self.transform = nn.Linear(self.encoder.bert.get_sentence_embedding_dimension() if encoder_type=="st" else configuration.hidden_size, self.transform_dim)
    
    def split_hf_config(self, s):
        digits = re.findall(r'\d+', s)
        nondigits = re.findall(r'\D+', s)
        return digits, nondigits

    def forward(self, input_ids, attention_mask):
        if(self.transform_dim > 0):
            return self.transform(self.encoder(input_ids,attention_mask))
        else:
            return self.encoder(input_ids, attention_mask)
        
    @property
    def repr_dims(self):
        return 768 if self.transform_dim < 0 else self.transform_dim