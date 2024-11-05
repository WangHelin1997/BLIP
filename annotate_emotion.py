# pip install -U funasr
from funasr import AutoModel
import argparse
import os
from tqdm import tqdm
from datasets import load_from_disk

emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'None', 'sad', 'surprised', 'None']
model = AutoModel(model="iic/emotion2vec_plus_large")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", default="/data/lmorove1/hwang258/newblip/dev_chunks_0000", type=str, help="dir saved wavform")
    parser.add_argument("--cache_dir", default="/data/lmorove1/hwang258/newblip/gigaspeech-val", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--save_path", default="/data/lmorove1/hwang258/newblip/gigaspeech-val-emotion.txt", type=str, help="If specified, save the dataset on disk with this path.")
    args = parser.parse_args()

    os.makedirs(args.save_path.rsplit('/',1)[0], exist_ok=True)

    hf_dataset = load_from_disk(args.cache_dir)
    save_list = []
    for index in tqdm(range(len(hf_dataset))):
    # for index in tqdm(range(50)):
        segment_id = hf_dataset["segment_id"][index]
        wav_path = os.path.join(args.wav_dir, segment_id+'.wav')
        if os.path.exists(wav_path):
            rec_result = model.generate(wav_path, output_dir="./outputs", granularity="utterance", extract_embedding=False)
            tmp = rec_result[0]['scores'].index(max(rec_result[0]['scores']))
            emotion_label = emotion_labels[tmp]
            save_list.append(segment_id+'\t'+emotion_label)

    with open(args.save_path, 'w') as file:
        for item in save_list:
            file.write(f"{item}\n")
