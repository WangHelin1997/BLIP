import json
import os
import random
import re
import torch
from torch.utils.data import Dataset
from data.utils import pre_caption
import glob
import numpy as np

class GigaSpeech(Dataset):
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
        hubert_fea = torch.from_numpy(np.load(os.path.join(self.fea_dir, segment_id+'.npz'))['arr_0'])
        return segment_id, hubert_fea, caption

if __name__ == '__main__':
    from datasets import load_from_disk
    hf_dataset = load_from_disk("/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train")
    fea_dir = "/data/lmorove1/hwang258/dataspeech/hubert_features"
    dataset = GigaSpeech(hf_dataset, fea_dir)
    print(len(dataset))
    print(dataset[4])
