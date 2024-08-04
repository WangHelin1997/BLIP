'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

# from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from models.audioencoder import CNNSelfAttention
from torch.nn.utils.rnn import pad_sequence

class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 input_dim = 25600,
                 hidden_dim = 512,
                 kernel_size = 5,
                 padding = 2,
                 pooling = 5,
                 dropout = 0.4,
                 audio_width = 512,                
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
        """               
        super().__init__()
        
        self.audio_encoder = CNNSelfAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            pooling=pooling,
            dropout=dropout,
            output_dim=audio_width
        )
        self.pooling = pooling
        
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = audio_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, audio=None, caption=None, mode='audio'):
        
        assert mode in ['audio', 'text', 'multimodal'], "mode parameter must be audio, text, or multimodal"
        
        if mode=='audio':   
            # return audio features
            features_len = torch.IntTensor([len(feat) for feat in audio]).to(device=audio.device)
            attention_mask = [
                torch.ones(math.ceil((l / self.pooling)))
                for l in features_len
            ]
            attention_mask = pad_sequence(attention_mask, batch_first=True)
            attention_mask = (1.0 - attention_mask) * -100000.0
            attention_mask = attention_mask.to(audio.device)
            audio_embeds = self.audio_encoder(audio, attention_mask) 
            return audio_embeds
        
        elif mode=='text':
            # return text features
            text = self.tokenizer(caption, return_tensors="pt").to(audio.device) 
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            text = self.tokenizer(caption, return_tensors="pt").to(audio.device) 
            features_len = torch.IntTensor([len(feat) for feat in audio]).to(device=audio.device)
            attention_mask = [
                torch.ones(math.ceil((l / self.pooling)))
                for l in features_len
            ]
            attention_mask = pad_sequence(attention_mask, batch_first=True)
            attention_mask = (1.0 - attention_mask) * -100000.0
            attention_mask = attention_mask.to(audio.device)
            
            audio_embeds = self.audio_encoder(audio, attention_mask) 
            audio_atts = torch.ones((audio_embeds.shape[0], 1), dtype=torch.long).to(audio.device)  
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = audio_embeds.unsqueeze(1),
                                       encoder_attention_mask = audio_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 input_dim = 25600,
                 hidden_dim = 512,
                 kernel_size = 5,
                 padding = 2,
                 pooling = 5,
                 dropout = 0.4,
                 audio_width = 512,
                 prompt = 'The description of this speech is:',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
        """            
        super().__init__()
        
        self.audio_encoder = CNNSelfAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            pooling=pooling,
            dropout=dropout,
            output_dim=audio_width
        )
        self.pooling = pooling
        
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = audio_width
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, audio, caption):

        features_len = torch.IntTensor([len(feat) for feat in audio]).to(device=audio.device)
        attention_mask = [
            torch.ones(math.ceil((l / self.pooling)))
            for l in features_len
        ]
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        attention_mask = (1.0 - attention_mask) * -100000.0
        attention_mask = attention_mask.to(audio.device)
        
        audio_embeds = self.audio_encoder(audio, attention_mask) 
        audio_atts = torch.ones((audio_embeds.shape[0], 1), dtype=torch.long).to(audio.device)  
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=50, return_tensors="pt").to(audio.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = audio_embeds.unsqueeze(1),
                                           encoder_attention_mask = audio_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        
        return loss_lm
        
    def generate(self, audio, sample=False, num_beams=3, max_length=50, min_length=10, top_p=0.9, repetition_penalty=1.0):
        
        features_len = torch.IntTensor([len(feat) for feat in audio]).to(device=audio.device)
        attention_mask = [
            torch.ones(math.ceil((l / self.pooling)))
            for l in features_len
        ]
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        attention_mask = (1.0 - attention_mask) * -100000.0
        attention_mask = attention_mask.to(audio.device)
        
        audio_embeds = self.audio_encoder(audio, attention_mask).unsqueeze(1)

        if not sample:
            audio_embeds = audio_embeds.repeat_interleave(num_beams,dim=0)
            
        audio_atts = torch.ones((audio_embeds.shape[0], 1), dtype=torch.long).to(audio.device) 
        model_kwargs = {"encoder_hidden_states": audio_embeds, "encoder_attention_mask":audio_atts}
        
        prompt = [self.prompt] * audio.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(audio.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
    

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
