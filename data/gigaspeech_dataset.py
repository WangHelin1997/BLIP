import os
import json
import random
import re
from torch.utils.data import Dataset
from data.utils import pre_caption
import glob
import torch

class GigaSpeech_train(Dataset):
    def __init__(self, hf_dataset, fea_dir, max_words=50):
        self.dataset = hf_dataset
        self.fea_dir = fea_dir
        self.max_words = max_words

        self.audio_ids = {}  
        n = 0
        for audio_id in self.dataset["segment_id"]:
            if audio_id not in self.audio_ids.keys():
                self.audio_ids[audio_id] = n
                n += 1 

    def __len__(self):
        
        return len(self.dataset)

    def clean(self, s):
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'"([^"]*)"', r'\1', s)
        s = re.sub(r"'([^']*)'", r'\1', s)
        return s
        
    def __getitem__(self, index): 
        tag = random.randint(1, 5)
        segment_id = self.dataset["segment_id"][index]
        caption = self.dataset["text_description"+str(tag)][index]
        caption = self.clean(caption)
        caption = pre_caption(caption, self.max_words)
        hubert_fea = torch.load(os.path.join(self.fea_dir, segment_id+'.pt'))
        return hubert_fea, caption, self.audio_ids[segment_id] 
    
class GigaSpeech_caption_eval(Dataset):
    def __init__(self, hf_dataset, fea_dir, max_words=50):  
        self.dataset = hf_dataset
        self.fea_dir = fea_dir
        self.max_words = max_words

    def __len__(self):
        
        return len(self.dataset)

    def clean(self, s):
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'"([^"]*)"', r'\1', s)
        s = re.sub(r"'([^']*)'", r'\1', s)
        return s
        
    def __getitem__(self, index): 
        tag = random.randint(1, 5)
        segment_id = self.dataset["segment_id"][index]
        caption = self.dataset["text_description"+str(tag)][index]
        caption = self.clean(caption)
        caption = pre_caption(caption, self.max_words)
        hubert_fea = torch.load(os.path.join(self.fea_dir, segment_id+'.pt'))
        return hubert_fea, segment_id
    
    
class GigaSpeech_retrieval_eval(Dataset):
    def __init__(self, hf_dataset, fea_dir, max_words=50):  
        self.dataset = hf_dataset
        self.fea_dir = fea_dir
        self.max_words = max_words
        
        self.text = []
        self.audio = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for idx in range(len(self.dataset)):
            segment_id = self.dataset["segment_id"][idx]
            self.audio.append(segment_id)
            self.img2txt[idx] = []
            for j in range(5):
                caption = self.dataset["text_description"+str(j+1)][idx]
                caption = self.clean(caption)
                caption = pre_caption(caption, self.max_words)
                self.text.append(caption)
                self.img2txt[idx].append(txt_id)
                self.txt2img[txt_id] = idx
                txt_id += 1
                                    
    def __len__(self):
        return len(self.dataset)

    def clean(self, s):
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'"([^"]*)"', r'\1', s)
        s = re.sub(r"'([^']*)'", r'\1', s)
        return s
    
    def __getitem__(self, index):    
        
        segment_id = self.dataset["segment_id"][index]
        hubert_fea = torch.load(os.path.join(self.fea_dir, segment_id+'.pt'))
        
        return hubert_fea, index