import torch
from datasets import load_dataset, load_from_disk
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import fairseq

def pad_or_trim(waveform, target_length):
    current_length = waveform.shape[1]

    if current_length > target_length:
        waveform = waveform[:, :target_length]
    elif current_length < target_length:
        pad_length = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    return waveform

# Define the dataset wrapper class
class StreamingAudioDataset(Dataset):
    def __init__(self, hf_dataset, target_length):
        self.dataset = hf_dataset
        self.target_length = target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): 
        segment_id = self.dataset[index]["segment_id"]
        audio = torch.FloatTensor(self.dataset[index]["audio"]['array']).unsqueeze(0)
        audio = pad_or_trim(audio, int(self.target_length * self.dataset[index]['audio']['sampling_rate'])).squeeze()
        return {"segment_id": segment_id, "audio": audio}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train", type=str, help="Cache dir to download data")
    parser.add_argument("--output_dir", default="/data/lmorove1/hwang258/dataspeech/hubert_features", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--target_length", default=5, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Download the pre-trained HuBERT model
    model_path = 'hubert_large_ll60k.pt'
    if not os.path.exists(model_path):
        os.system(f'wget https://dl.fbaipublicfiles.com/hubert/{model_path}')
    
    # Load the pre-trained HuBERT model using Fairseq
    hubert_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = hubert_model[0].to(device)
    model.eval()

    # Use DataParallel for multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_from_disk(args.cache_dir)
    streaming_audio_dataset = StreamingAudioDataset(dataset, args.target_length)
    
    # Extend dataset
    extended_dataset = torch.utils.data.ConcatDataset([streaming_audio_dataset] * 100)
    
    # Increase number of workers and pin memory
    dataloader = DataLoader(extended_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            audio = batch['audio'].to(device, non_blocking=True)
            features = model.module.extract_features(audio, padding_mask=None)
            reps = features[0].cpu()
            
            for idx, segment_id in enumerate(batch['segment_id']):
                torch.save(reps[idx], os.path.join(args.output_dir, segment_id+'.pt'))
