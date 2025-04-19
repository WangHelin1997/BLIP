import os
import json
import random
import re
from torch.utils.data import Dataset
from data.utils import pre_caption
import glob
import torch
import numpy as np
from tqdm import tqdm

class Speech_train(Dataset):
    def __init__(self, hf_dataset, config, max_words=50):
        self.dataset = hf_dataset
        self.config = config
        self.max_words = max_words

        self.audio_ids = {}  
        self.captions = {}
        self.audio_paths = []

        n = 0
        for audio_path, source, caption in tqdm(zip(self.dataset["audio_path"], self.dataset["source"], self.dataset["caption"])):
            if source == "libritts-r":
                audio_path = os.path.join(config['librittsr_dir'], audio_path).replace(".wav",".npz")
            else:
                audio_path = os.path.join(config['other_dir'], audio_path).replace(".wav",".npz")
            if os.path.exists(audio_path):
                self.audio_paths.append(audio_path)
                self.audio_ids[audio_path] = n
                self.captions[audio_path] = caption
                n += 1 

    def __len__(self):
        
        return len(self.audio_ids)

    def clean(self, s):
        s = re.sub(r'\(.*?\)', '', s)
        s = re.sub(r'"([^"]*)"', r'\1', s)
        s = re.sub(r"'([^']*)'", r'\1', s)
        return s
        
    def __getitem__(self, index): 
        segment_id = self.audio_paths[index]
        caption = self.captions[segment_id]
        caption = self.clean(caption)
        caption = pre_caption(caption, self.max_words)
        hubert_fea = torch.from_numpy(np.load(segment_id)['arr_0'])
        return hubert_fea, caption, self.audio_ids[segment_id] 
    
class Speech_caption_eval(Dataset):
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
        # hubert_fea = torch.from_numpy(np.load(os.path.join(self.fea_dir, segment_id+'.npz'))['arr_0'])
        hubert_fea = torch.from_numpy(self.fea_dir[segment_id][:])
        return hubert_fea, segment_id
    
    
class Speech_retrieval_eval(Dataset):
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
        # hubert_fea = torch.from_numpy(np.load(os.path.join(self.fea_dir, segment_id+'.npz'))['arr_0'])
        hubert_fea = torch.from_numpy(self.fea_dir[segment_id][:])
        
        return hubert_fea, index
