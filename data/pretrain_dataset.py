import json
import os
import random
import re
import torch
from torch.utils.data import Dataset
from data.utils import pre_caption
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class Speech_Pretrain(Dataset):
    def __init__(self, hf_dataset, config, max_words=50, num_threads=8):
        self.dataset = hf_dataset
        self.config = config
        self.max_words = max_words

        self.audio_ids = {}  
        self.captions = {}
        self.audio_paths = []

        self._process_all(num_threads)

    def _process_entry(self, args):
        idx, audio_path, source, caption = args
        if source == "libritts-r":
            path = os.path.join(self.config['librittsr_dir'], audio_path).replace(".wav", ".npz")
        else:
            path = os.path.join(self.config['other_dir'], audio_path).replace(".wav", ".npz")

        if os.path.exists(path):
            return (idx, path, caption)
        return None

    def _process_all(self, num_threads):
        data = list(enumerate(zip(self.dataset["audio_path"], self.dataset["source"], self.dataset["caption"])))
        args = [(i, audio_path, source, caption) for i, (audio_path, source, caption) in data]

        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for result in tqdm(executor.map(self._process_entry, args), total=len(args)):
                if result is not None:
                    results.append(result)

        for n, path, caption in results:
            self.audio_paths.append(path)
            self.audio_ids[path] = n
            self.captions[path] = caption


    def __len__(self):
        
        return len(self.audio_paths)

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
        return segment_id, hubert_fea, caption

if __name__ == '__main__':
    from datasets import load_from_disk
    hf_dataset = load_from_disk("/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train")
    fea_dir = "/data/lmorove1/hwang258/dataspeech/hubert_features"
    dataset = GigaSpeech(hf_dataset, fea_dir)
    print(len(dataset))
    print(dataset[4])

