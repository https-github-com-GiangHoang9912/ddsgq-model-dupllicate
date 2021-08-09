import torch
import torch.nn as nn
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from pytorch_transformers import BertModel
import re

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SBert(nn.Module):
    def __init__(self, sbert: SentenceTransformer, trainable: bool=False):
        super(SBert, self).__init__()

        i = 0
        for child in sbert.children():
            if i == 0:
                self.transformer = child
            else:
                self.pooling = child

            i += 1

        
        for param in self.transformer.parameters():
            param.requires_grad = trainable
        for param in self.pooling.parameters():
            param.requires_grad = trainable

    def forward(self, features):
        output = self.transformer(features)
        output = self.pooling(output)
        output = output["sentence_embedding"]

        return output

class TBert(nn.Module):
    def __init__(self,bert: BertModel, lstm_hidden_size: int=384, trainable: bool=False):
        super(TBert, self).__init__()
        
        self.embeddings = bert
        
        i = 0
        for child in self.embeddings.children():
            if i < 2:
                for param in child.parameters():
                    param.requires_grad = trainable
            i += 1

    def forward(self, features):
        output, __ = self.embeddings(features["input_ids"],features["attention_mask"],features["attention_mask"])

        att_mask = features["attention_mask"].unsqueeze(-1).expand(output.shape).float()
        output = output*att_mask

        return torch.max(output, 1)[0]

class Output_Layer(nn.Module):
    def __init__(self, num_classes: int, input_size: int, hidden_size: int=256, dropout: float=0.1):
        super(Output_Layer, self).__init__()

        self.tbert_norm = LayerNorm(hidden_size)

        self.f_linear = nn.Linear(input_size,hidden_size)
        self.dropout = nn.Dropout(0.1)

        self.o_linear = nn.Linear(hidden_size,num_classes)

        self.sigmoid = nn.Sigmoid()

        self.f_conection = nn.Linear(num_classes*2, hidden_size//4)
        self.o_conection = nn.Linear(hidden_size//4, num_classes)
        
    def forward(self, inputI, inputII):
        sbert_score = torch.cosine_similarity(inputI[0],inputII[0]).unsqueeze(-1)

        tbert_maxtrix = torch.cat([inputI[1],inputII[1]],dim=-1)

        tbert_out = self.f_linear(tbert_maxtrix)
        tbert_norm = self.tbert_norm(tbert_out)
        tbert_out = torch.relu(tbert_norm)
        tbert_out = self.dropout(tbert_out)

        tbert_out = self.o_linear(tbert_out)
        tbert_out = torch.tanh(tbert_out)
        
        conection_output = torch.cat([tbert_out,sbert_score],dim=-1)
        conection_output = self.f_conection(conection_output)
        conection_output = torch.relu(conection_output)
        conection_output = self.o_conection(conection_output)
        
        return (self.sigmoid(conection_output)+sbert_score)/2, tbert_out


class Bert_Model(nn.Module):
    def __init__(self, sbert: SentenceTransformer, bert: BertModel, lstm_hidden_size: int=768):
        super(Bert_Model, self).__init__()
        self.tokenizer = sbert.tokenizer

        self.sbert = SBert(sbert, trainable=False)
        self.tbert = TBert(bert, lstm_hidden_size, trainable=False)

        self.output_layer = Output_Layer(1, lstm_hidden_size*2, 256)

    def forward(self, features1, features2):
        outputI = self.encode(features1)
        outputII = self.encode(features2)

        return self.output_layer(outputI, outputII)
    
    def encode(self, features):
        s_output = self.sbert(features)
        t_output = self.tbert(features)

        return s_output, t_output
    
    def tokenize(self, text):
        return self.tokenizer(text)


class SBertTokenizer():
    def __init__(self, model=None):
        self.tokenizer = model.tokenizer
        self.lemmatizer = WordNetLemmatizer()

    def _truncate_seq(self, ids, att, max_length: int):
        while True:
            ids.pop()
            att.pop()
            total_length = len(ids)
            if total_length <= max_length:
                break

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]','',text, re.UNICODE)
        text = text.lower()
        text = [self.lemmatizer.lemmatize(token) for token in text.split(" ")]
        text = [self.lemmatizer.lemmatize(token, "v") for token in text]
        text = " ".join(text)
        return text

    def tokenize(self, data, max_sequence_length: int=64):
        ids_box = []
        att_box = []
        for sentence in tqdm(data):
            ids = []
            att = []
            sentence = self.clean_text(sentence)
            if isinstance(sentence, str) and len(sentence) > 0:           
                cl = [sentence]
                sentence_tokenized = self.tokenizer(cl)
                ids = sentence_tokenized["input_ids"][0]
                att = sentence_tokenized["attention_mask"][0]
                l_ids = ids[-1]
                l_att = att[-1]
                self._truncate_seq(ids,att,max_sequence_length-1) 
                ids.append(l_ids)
                att.append(l_att)
                while len(ids) < max_sequence_length:
                    ids.append(0)
                    att.append(0)
            else:
                while len(ids) < max_sequence_length:
                    ids.append(0)
                    att.append(0)

            ids_box.append(ids)
            att_box.append(att)    

        dataset_input_ids = torch.tensor(ids_box, dtype=torch.long)
        dataset_attention_masks = torch.tensor(att_box, dtype=torch.long)

        features = {"input_ids": dataset_input_ids, "attention_mask": dataset_attention_masks}
        return features