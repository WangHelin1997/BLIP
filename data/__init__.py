import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from data.pretrain_dataset import Speech_Pretrain
from data.speech_dataset import Speech_train, Speech_caption_eval, Speech_retrieval_eval

def create_dataset(dataset, config):     
    if dataset=='pretrain':
        hf_dataset = load_dataset(config["dataset_name"])["train"]
        dataset = Speech_Pretrain(hf_dataset, config, max_words=config["max_words"]) 

        return dataset  
    
    elif dataset=='caption':     
        hf_dataset = load_dataset(config["dataset_name"])
        train_dataset = Speech_train(hf_dataset["train"], config, max_words=config["max_words"]) 
        val_dataset = Speech_caption_eval(hf_dataset["val"], config, max_words=config["max_words"]) 
        test_dataset = Speech_caption_eval(hf_dataset["test"], config, max_words=config["max_words"])    
        
        return train_dataset, val_dataset, test_dataset 
        
    elif dataset=='retrieval': 
        train_dataset = Speech_train(hf_dataset["train"], config, max_words=config["max_words"]) 
        val_dataset = Speech_retrieval_eval(hf_dataset["val"], config, max_words=config["max_words"]) 
        test_dataset = Speech_retrieval_eval(hf_dataset["test"], config, max_words=config["max_words"])  
        
        return train_dataset, val_dataset, test_dataset 
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

