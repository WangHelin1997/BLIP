import torch
# from datasets import load_dataset, load_from_disk
import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import fairseq
import glob
import librosa
import numpy as np

sr = 16000

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
    def __init__(self, wav_dir, start, end, target_length):
        files = glob.glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True)
        self.files = files[start:end]
        self.target_length = target_length
        self.wav_dir = wav_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index): 
        segment_id = self.files[index].replace(self.wav_dir, '').split('.wav')[0]
        audio, _ = librosa.load(self.files[index], sr=sr)
        audio = torch.FloatTensor(audio).unsqueeze(0)
        audio = pad_or_trim(audio, int(self.target_length * sr)).squeeze()
        return {"segment_id": segment_id, "audio": audio}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", default="/export/corpora7/LibriTTS_R/", type=str, help="dir saved wavform")
    parser.add_argument("--output_dir", default="/export/corpora7/CapSpeech-real/LibriTTS_R_hubert_features/", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--target_length", default=10, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=1000000, type=int)
    
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

    os.makedirs(args.output_dir, exist_ok=True)

    streaming_audio_dataset = StreamingAudioDataset(args.wav_dir, args.start, args.end, args.target_length)
    
    # Increase number of workers and pin memory
    dataloader = DataLoader(streaming_audio_dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            audio = batch['audio'].to(device, non_blocking=True)
            features = model.extract_features(audio, padding_mask=None)
            reps = features[0].cpu()
            
            for idx, segment_id in enumerate(batch['segment_id']):
                savepath = os.path.join(args.output_dir, segment_id+'.npz')
                os.makedirs(os.path.dirname(savepath), exist_ok=True)
                np.savez_compressed(savepath, reps[idx].numpy())
