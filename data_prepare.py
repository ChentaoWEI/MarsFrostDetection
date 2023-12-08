import logging
from pathlib import Path
import os
import random

def load_subdir_data(dir_path, image_size, seed=None): 
    # Grab only the classes that (1) we want to keep and (2) exist in this directory
    tile_dir = dir_path / Path('tiles')
    label_dir = dir_path / Path('labels')
    
    loc_list = []
    
    for folder in os.listdir(tile_dir):
        if os.path.isdir(os.path.join(tile_dir, folder)):
            for file in os.listdir(os.path.join(tile_dir, folder)):
                if file.endswith(".png"):
                    loc_list.append((os.path.join(os.path.join(tile_dir, folder), file), folder))

    return loc_list

def prepare_data():
    IMAGE_SIZE = (299, 299)  # All images contained in this dataset are 299x299 (originally, to match Inception v3 input size)
    SEED = 17

    # Head directory containing all image subframes. Update with the relative path of your data directory
    data_head_dir = Path('../data')

    # Find all subframe directories
    subdirs = [Path(subdir.stem) for subdir in data_head_dir.iterdir() if subdir.is_dir()]
    src_image_ids = ['_'.join(a_path.name.split('_')[:3]) for a_path in subdirs]

    # Load train/val/test subframe IDs
    def load_text_ids(file_path):
        """Simple helper to load all lines from a text file"""
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    # Load the subframe names for the three data subsets
    train_ids = load_text_ids('./train_source_images.txt')
    validate_ids = load_text_ids('./val_source_images.txt')
    test_ids = load_text_ids('./test_source_images.txt')

    # Generate a list containing the dataset split for the matching subdirectory names
    subdir_splits = []
    for src_id in src_image_ids:
        if src_id in train_ids:
            subdir_splits.append('train')
        elif src_id in validate_ids:
            subdir_splits.append('validate')
        elif(src_id in test_ids):
            subdir_splits.append('test')
        else:
            logging.warning(f'{src_id}: Did not find designated split in train/validate/test list.')
            subdir_splits.append(None)
    
    data_train, data_test, data_val = [], [], []
    for subdir, split in zip(subdirs, subdir_splits):
        full_path = data_head_dir / subdir
        if split=='validate':
            data_val.extend(load_subdir_data(full_path, IMAGE_SIZE, SEED))
        elif split=='train':
            data_train.extend(load_subdir_data(full_path, IMAGE_SIZE, SEED))
        elif split=='test':
            data_test.extend(load_subdir_data(full_path, IMAGE_SIZE, SEED))
    # random.shuffle(data_train)
    # random.shuffle(data_test)
    # random.shuffle(data_val)
    
    img_list_train, label_list_train = zip(*data_train)
    img_list_test, label_list_test = zip(*data_test)
    img_list_val, label_list_val = zip(*data_val)
    
    return img_list_train, label_list_train, img_list_test, label_list_test, img_list_val, label_list_val
