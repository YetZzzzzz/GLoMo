import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


idx_to_label = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'suprise',
}


labels_en = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
labels_ch = ['愤怒', '厌恶', '恐惧', '高兴', '平静', '悲伤', '惊奇']


class MMSAATBaselineDataset(Dataset):
    def __init__(self, stage):
        
        self.stage = stage
        self.dataset_path = './datasets/CHERMA0723/' + self.stage + '.json'
        
        self.filename_label_list = []

        with open(self.dataset_path) as f:
            for example in json.load(f):
                a = example['audio_file'].replace('.wav', '')
                v = example['video_file']
                self.filename_label_list.append((a, v, example['txt_label'], example['audio_label'], example['visual_label'], example['video_label']))
                
    def __len__(self):
        return len(self.filename_label_list)

    def __getitem__(self, idx):
        current_filename, current_filename_v, label_t, label_a, label_v, label_m = self.filename_label_list[idx]
        
        text_vector = np.load('./datasets/CHERMA0723/text/' + self.stage + '/' + current_filename + '.npy')
        text_vector = torch.from_numpy(text_vector)

        video_vector = np.load('./datasets/CHERMA0723/visual/' + self.stage + '/' + current_filename + '.mp4.npy')
        video_vector = torch.from_numpy(video_vector)
        ## here pad the visual lengths
        
        audio_vector = np.load('./datasets/CHERMA0723/audio/'+ self.stage + '/' + current_filename + '.npy') 
        audio_vector = torch.from_numpy(audio_vector)

        return  text_vector, audio_vector, video_vector, labels_ch.index(label_t), labels_ch.index(label_a), labels_ch.index(label_v), labels_ch.index(label_m)
        
        