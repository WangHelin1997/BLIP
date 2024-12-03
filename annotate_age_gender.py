import audeer
import audonnx
import librosa
import argparse
import numpy as np
import os
from tqdm import tqdm
from datasets import load_from_disk
from multiprocessing import Pool, Manager, cpu_count

age_labels = ['child', 'teenager', 'young adult', 'middle-aged adult', 'elderly']
gender_labels = ['female', 'male']
url = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
cache_root = audeer.mkdir('cache')
model_root = audeer.mkdir('model')
sampling_rate = 16000
archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)


def process_item(index, hf_dataset, args):
    """Process a single dataset item."""
    segment_id = hf_dataset["segment_id"][index]
    wav_path = os.path.join(args.wav_dir, segment_id + '.wav')
    if os.path.exists(wav_path):
        signal, _ = librosa.load(wav_path, sr=sampling_rate)
        result = model(signal, sampling_rate)

        # Process age
        age_label = result['logits_age'].squeeze() * 100.0
        if age_label <= 12:
            age_label = 'child'
        elif age_label <= 19:
            age_label = 'teenager'
        elif age_label <= 39:
            age_label = 'young adult'
        elif age_label <= 64:
            age_label = 'middle-aged adult'
        else:
            age_label = 'elderly'

        # Process gender
        gender_label = result['logits_gender'].squeeze()
        gender_label = gender_label[:2]  # Remove child
        gender_label = np.argmax(gender_label)

        return segment_id, age_label, gender_labels[gender_label]
    return None


def worker_fn(indices, hf_dataset, args, shared_list_age, shared_list_gender):
    """Worker function for processing in parallel."""
    for index in tqdm(indices):
        result = process_item(index, hf_dataset, args)
        if result:
            segment_id, age_label, gender_label = result
            shared_list_age.append(f"{segment_id}\t{age_label}")
            shared_list_gender.append(f"{segment_id}\t{gender_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", default="/data/lmorove1/hwang258/SSR-Speech/gigacaps-val-processed/audios", type=str, help="dir saved wavform")
    parser.add_argument("--cache_dir", default="/data/lmorove1/hwang258/SSR-Speech/gigacaps-val", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--save_path_age", default="/data/lmorove1/hwang258/giga-debug-age.txt", type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--save_path_gender", default="/data/lmorove1/hwang258/giga-debug-gender.txt", type=str, help="If specified, save the dataset on disk with this path.")
    args = parser.parse_args()

    os.makedirs(args.save_path_age.rsplit('/', 1)[0], exist_ok=True)
    os.makedirs(args.save_path_gender.rsplit('/', 1)[0], exist_ok=True)

    # Load dataset
    hf_dataset = load_from_disk(args.cache_dir)['train']

    # Prepare multiprocessing
    num_processes = cpu_count()
    indices = list(range(len(hf_dataset)))
    chunk_size = len(indices) // num_processes

    # Shared lists to store results
    with Manager() as manager:
        shared_list_age = manager.list()
        shared_list_gender = manager.list()

        # Create pool
        with Pool(processes=num_processes) as pool:
            pool.starmap(
                worker_fn,
                [
                    (indices[i:i + chunk_size], hf_dataset, args, shared_list_age, shared_list_gender)
                    for i in range(0, len(indices), chunk_size)
                ]
            )

        # Save results
        with open(args.save_path_age, 'w') as file:
            for item in shared_list_age:
                file.write(f"{item}\n")

        with open(args.save_path_gender, 'w') as file:
            for item in shared_list_gender:
                file.write(f"{item}\n")
