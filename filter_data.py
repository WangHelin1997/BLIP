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
MODEL = "openai/whisper-large-v3-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL)
processor = WhisperProcessor.from_pretrained(MODEL, language="en", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(MODEL, language="en", task="transcribe")
whisper_model = WhisperForConditionalGeneration.from_pretrained(MODEL).to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
# clean up the gigaspeech transcripts
punc2sym = {" <COMMA>": ",", " <PERIOD>": ".", " <QUESTIONMARK>": "?", " <EXCLAMATIONPOINT>": "!"} # note the space in front of each punc name
forbidden_words = ["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"]


def asr(segment_id, wav, text):
    for k, v in punc2sym.items():
        text = text.replace(k, v)
    if sum(word in forbidden_words for word in text.split(" ")):
        print(f"skip {segment_id}, because it contains forbiden words. It's transcript: {text}")
        return None, None
    mel = feature_extractor(wav.squeeze(0).cpu().numpy(), sampling_rate=16_000, return_tensors="pt")['input_features']
    generated_ids = whisper_model.generate(inputs=mel.to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return pred, text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", default="/data/lmorove1/hwang258/newblip/dev_chunks_0000", type=str, help="dir saved wavform")
    parser.add_argument("--cache_dir", default="/data/lmorove1/hwang258/newblip/gigaspeech-val", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--save_path", default="/data/lmorove1/hwang258/newblip/gigaspeech-val.txt", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--audio_min_length", default=2., type=float)
    parser.add_argument("--text_min_length", default=10, type=int)
    parser.add_argument("--min_wer", default=0.3, type=float)
    args = parser.parse_args()

    os.makedirs(args.save_path.rsplit('/',1)[0], exist_ok=True)

    hf_dataset = load_from_disk(args.cache_dir)
    save_list = []
    # for index in tqdm(range(len(hf_dataset))):
    for index in tqdm(range(50)):
        segment_id = hf_dataset["segment_id"][index]
        wav_path = os.path.join(args.wav_dir, segment_id+'.wav')
        if os.path.exists(wav_path):
            audio, sr = torchaudio.load(wav_path)
            if audio.shape[-1] / sr >= args.audio_min_length:
                trans = hf_dataset["text"][index]
                if len(trans.split(' ')) >= args.text_min_length:
                    pred, gt = asr(segment_id, audio, trans)
                    forward_asr = normalizer(pred)
                    forward_gt = normalizer(gt)
                    if forward_asr is not None:
                        forward_wer = round(calculate_wer([forward_gt if len(forward_gt) > 0 else '<UNK>'], [forward_asr if len(forward_asr) > 0 else '<UNK>']), 3)
                        if forward_wer <= args.min_wer:
                            save_list.append(segment_id+'\t'+gt+'\t'+pred+'\t'+str(forward_wer))

    with open(args.save_path, 'w') as file:
        for item in save_list:
            file.write(f"{item}\n")
