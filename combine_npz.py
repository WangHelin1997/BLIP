import os
import numpy as np
import h5py
import glob
from tqdm import tqdm

def load_npz_files(directory):
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    data = {}
    
    for file in tqdm(npz_files):
        seg_id = file.split('/')[-1].split('.npz')[0]
        data[seg_id] = np.load(file)['arr_0']
                
    return data

def save_to_hdf5(data, output_file):
    with h5py.File(output_file, 'w') as hdf5_file:
        for key, value in data.items():
            hdf5_file.create_dataset(key, data=value)

if __name__ == "__main__":
    # Directory containing the .npz files
    directory = '/data/lmorove1/hwang258/dataspeech/hubert_features'
    # Output HDF5 file
    output_file = '/data/lmorove1/hwang258/dataspeech/hubert_features.h5'
    
    # Load .npz files
    data = load_npz_files(directory)
    
    # Save to HDF5
    save_to_hdf5(data, output_file)
    
    print(f"Data has been successfully written to {output_file}")
