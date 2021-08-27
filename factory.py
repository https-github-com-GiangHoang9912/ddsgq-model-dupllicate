from model.model import SBertTokenizer
from question import Question
import shutil
import torch
import pandas as pd
import os

dirname, filename = os.path.split(os.path.abspath(__file__))

class Factory():
    def __init__(self, tokenizer=SBertTokenizer, model_path=os.path.join(dirname,"model/sbert.model.ep4"), with_cuda: bool=True):
        self.cuda_condition = torch.cuda.is_available() and with_cuda

        self.device = torch.device("cuda:0" if self.cuda_condition else "cpu")
        self.load_model(model_path)
        
        self.tokenizer = tokenizer(self.model)

        self.data = None

        self.load_encoded()

    def tokenize(self, data: list):
        return self.tokenizer.tokenize(data)

    def encode(self, data: list):
        features = self.tokenize(data)
        with torch.no_grad():
            features = {k:v.to(self.device) for k,v in features.items()}
            output = self.model.encode(features)
        
        return output
    
    def save_encoded(self, source_path=os.path.join(dirname,"train.csv"), destination_path=os.path.join(dirname,"train.pt")):
        df = pd.read_csv(source_path)
        data = df['sentence'].values.tolist()

        self.data[0] += data
        encoded = self.encode(self.data[0])

        box = [self.data[0],encoded]

        # overwrite train.pt
        # encoded = self.encode(data)
        # box = [data,encoded]

        torch.save(box,destination_path)

    def save_encoded_sentence(self, sentences, destination_path=os.path.join(dirname,"train.pt")):
        self.data[0].append(sentences.question)
        encoded = self.encode(self.data[0])
        box = [self.data[0],encoded]
        
        torch.save(box, destination_path)

    def save_encoded_with_subject(self, dataRequest):
        src = os.path.join(dirname,"train.pt")
        destination_path =  os.path.join(dirname + "/subject/", dataRequest['subject'] + ".pt")
        
        if not os.path.isfile(destination_path):
            shutil.copyfile(src, destination_path)

        print(destination_path)

        if os.path.isfile(destination_path):
            with open(destination_path,"rb") as f:
                data = torch.load(f,map_location=torch.device('cpu'))
                
                if len(data[0]) != len(dataRequest['dataBank']):
                    encoded = self.encode(dataRequest['dataBank'])
                    box = [dataRequest['dataBank'],encoded]
                    torch.save(box, destination_path)


        
    def load_encoded(self, path=os.path.join(dirname,"train.pt")):
        if os.path.isfile(path):
            with open(path,"rb") as f:
                self.data = torch.load(f,map_location=torch.device('cpu'))
        else:
            print("encoded path not exist !!!")

    def load_model(self, model_path):
        if os.path.isfile(model_path):
            with open(model_path,"rb") as f:
                self.model = torch.load(f)
            self.model.to(self.device)
        else:
            print("model path not exist !!!")

    def find_duplicate(self, input_data: list, confident=0.8, subject="train"):
        self.load_encoded(path=os.path.join(dirname + "/subject/", subject + ".pt"))
        if self.data is not None:
            output = self.encode(input_data)
            output = [o.expand(self.data[1][0].size()) for o in output]

            scores, __ = self.model.output_layer(self.data[1],output) 
            scores = scores.squeeze(-1)

            scores = {self.data[0][idx]:scores[idx].item() for idx in range(len(scores))}
            scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1],reverse=True)}
            
            response = []
            for k, v in sorted(scores.items(), key=lambda item: item[1],reverse=True):
                response.append(Question(k, v))

            fit = {}
            for k,v in scores.items():
                if v >= confident:
                    fit[k] = v
            return fit, response[0:3]
        else:
            print("data is empty !!!")