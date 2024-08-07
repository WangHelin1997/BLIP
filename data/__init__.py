import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from data.pretrain_dataset import GigaSpeech
from data.gigaspeech_dataset import GigaSpeech_train, GigaSpeech_caption_eval, GigaSpeech_retrieval_eval
import h5py

def create_dataset(dataset, config): 
    features = h5py.File(config['fea_dir'], 'r')
    
    if dataset=='pretrain':
        hf_dataset = load_from_disk(config['pretrain_cache_dir'])
        # dataset = GigaSpeech(hf_dataset, config['fea_dir']) 
        dataset = GigaSpeech(hf_dataset, features) 
        # dataset = torch.utils.data.ConcatDataset([dataset] * 40000)

        return dataset  
    
    elif dataset=='caption':   
        # train_dataset = GigaSpeech_train(load_from_disk(config['caption_train_cache_dir']), config['fea_dir']) 
        # val_dataset = GigaSpeech_caption_eval(load_from_disk(config['caption_val_cache_dir']), config['fea_dir']) 
        # test_dataset = GigaSpeech_caption_eval(load_from_disk(config['caption_test_cache_dir']), config['fea_dir'])    
        train_dataset = GigaSpeech_train(load_from_disk(config['caption_train_cache_dir']), features) 
        val_dataset = GigaSpeech_caption_eval(load_from_disk(config['caption_val_cache_dir']), features) 
        test_dataset = GigaSpeech_caption_eval(load_from_disk(config['caption_test_cache_dir']), features)    
        
        return train_dataset, val_dataset, test_dataset 
        
    elif dataset=='retrieval': 
        # train_dataset = GigaSpeech_train(load_from_disk(config['caption_train_cache_dir']), config['fea_dir']) 
        # val_dataset = GigaSpeech_retrieval_eval(load_from_disk(config['caption_val_cache_dir']), config['fea_dir']) 
        # test_dataset = GigaSpeech_retrieval_eval(load_from_disk(config['caption_test_cache_dir']), config['fea_dir'])  
        train_dataset = GigaSpeech_train(load_from_disk(config['caption_train_cache_dir']), features) 
        val_dataset = GigaSpeech_retrieval_eval(load_from_disk(config['caption_val_cache_dir']), features) 
        test_dataset = GigaSpeech_retrieval_eval(load_from_disk(config['caption_test_cache_dir']), features)  
        
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

