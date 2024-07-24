import s3prl.hub as hub
import torch
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def pad_or_trim(waveform, target_length):
    current_length = waveform.shape[1]

    if current_length > target_length:
        waveform = waveform[:, :target_length]
    elif current_length < target_length:
        pad_length = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    return waveform
    
# Define the dataset wrapper class
class StreamingAudioDataset(torch.utils.data.IterableDataset):
    def __init__(self, hf_dataset, target_length):
        self.dataset = hf_dataset
        self.target_length = target_length

    def __iter__(self):
        for item in self.dataset['train']:
            segment_id = item["segment_id"]
            audio = torch.FloatTensor(item["audio"]['array']).unsqueeze(0)
            audio = pad_or_trim(audio, int(self.target_length*item['audio']['sampling_rate'])).squeeze()
            yield {"segment_id": segment_id, "audio": audio}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="westbrook/gigaspeech-tiny-stage4")
    parser.add_argument("--configuration", default="default", type=str, help="Dataset configuration to use, if necessary.")
    parser.add_argument("--cache_dir", default="/data/lmorove1/hwang258/dataspeech/cache", type=str, help="Cache dir to download data")
    parser.add_argument("--output_dir", default="/data/lmorove1/hwang258/dataspeech/hubert_features", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--target_length", default=5, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    
    args = parser.parse_args()
    model = getattr(hub, 'hubert_large_ll60k')()  # build the Wav2Vec 2.0 model with pre-trained weights

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # or cpu
    model = model.to(device)
    wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
    dataset = load_dataset(args.dataset_name, args.configuration, cache_dir=args.cache_dir, streaming=True)
    # Wrap the dataset
    streaming_audio_dataset = StreamingAudioDataset(dataset, args.target_length)
    # Create DataLoader and set num_workers
    dataloader = DataLoader(streaming_audio_dataset, batch_size=args.batch_size, num_workers=1)
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            reps = model(batch['audio'].to(device))["hidden_states"]
            reps = torch.cat(reps, -1).cpu()
            for idx, segment_id in enumerate(batch['segment_id']):
                torch.save(reps[idx], os.path.join(args.output_dir, segment_id+'.pt'))