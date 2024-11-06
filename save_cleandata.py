from datasets import load_from_disk

cache_dir = "/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train"
target_path = "/data/lmorove1/hwang258/sc/sc/BLIP/output/gigaspeech-tiny-train.txt"
save_dir = "/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-clean"

savenames = []
with open(target_path, 'r') as file:
    for line in file:
        savenames.append(line.split('\t')[0])
        
hf_dataset = load_from_disk(cache_dir)
filtered_dataset = hf_dataset.filter(lambda example: example['segment_id'] in savenames)

filtered_dataset.save_to_disk(save_dir)
