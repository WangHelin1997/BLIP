from jiwer import wer as calculate_wer
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from whisper.normalizers import EnglishTextNormalizer
import argparse
import os
from tqdm import tqdm
import torchaudio
import torch
from datasets import load_from_disk

normalizer = EnglishTextNormalizer()
MODEL = "openai/whisper-medium.en"
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
processor = WhisperProcessor.from_pretrained(MODEL, language="en", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(MODEL, language="en", task="transcribe")
whisper_model = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

def asr(wav, text):
    mel = feature_extractor(wav.squeeze(0).cpu().numpy(), sampling_rate=16_000, return_tensors="pt")['input_features']
    generated_ids = whisper_model.generate(inputs=mel.to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pred = normalizer(generated_text)
    gt = normalizer(text)
    return pred, gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", default="/data/lmorove1/hwang258/dataspeech/cache/audios/", type=str, help="dir saved wavform")
    parser.add_argument("--cache_dir", default="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--save_path", default="/data/lmorove1/hwang258/sc/sc/BLIP/output/gigaspeech-tiny-train.txt", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--audio_min_length", default=2., type=float)
    parser.add_argument("--text_min_length", default=10, type=int)
    parser.add_argument("--min_wer", default=0.3, type=float)
    args = parser.parse_args()

    os.makedirs(args.save_path.rsplit('/',1)[0], exist_ok=True)

    hf_dataset = load_from_disk(args.cache_dir)
    save_list = []
    for index in tqdm(range(len(hf_dataset))):
        segment_id = hf_dataset["segment_id"][index]
        wav_path = os.path.join(args.wav_dir, segment_id+'.wav')
        if os.path.exists(wav_path):
            audio, sr = torchaudio.load(wav_path)
            if audio.shape[-1] / sr >= args.audio_min_length:
                trans = hf_dataset["text"][index]
                if len(trans.split(' ')) >= args.text_min_length:
                    forward_asr, forward_gt = asr(audio, trans)
                    forward_wer = round(calculate_wer([forward_gt if len(forward_gt) > 0 else '<UNK>'], [forward_asr if len(forward_asr) > 0 else '<UNK>']), 3)
                    if forward_wer <= args.min_wer:
                        save_list.append(segment_id)

    with open(args.save_path, 'w') as file:
        for item in save_list:
            file.write(f"{item}\n")
