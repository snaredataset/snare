import os
import json
import torch
import torch.utils.data

import numpy as np
import gzip
import json

class CLIPGraspingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode='train'):
        self.total_views = 14
        self.cfg = cfg
        self.mode = mode
        self.folds = os.path.join(self.cfg['data']['amt_data'], self.cfg['data']['folds'])
        self.feats_backbone = self.cfg['train']['feats_backbone']

        self.load_entries()
        self.load_extracted_features()

    def load_entries(self):
        train_train_files = ["train.json"]
        train_val_files = ["val.json"]
        test_test_files = ["test.json"]

        # modes
        if self.mode == "train":
            self.files = train_train_files
        elif self.mode  == 'valid':
            self.files = train_val_files
        elif self.mode == "test":
            self.files =  test_test_files
        else:
            raise RuntimeError('mode not recognized, should be train, valid or test: ' + str(self.mode))

        # load amt data
        self.data = []
        for file in self.files:
            fname_rel = os.path.join(self.folds, file)
            print(fname_rel)
            with open(fname_rel, 'r') as f:
                self.data = self.data + json.load(f)

        print(f"Loaded Entries. {self.mode}: {len(self.data)} entries")

    def load_extracted_features(self):
        if self.feats_backbone == "clip":
            lang_feats_path = self.cfg['data']['clip_lang_feats']
            with gzip.open(lang_feats_path, 'r') as f:
                self.lang_feats = json.load(f)

            img_feats_path = self.cfg['data']['clip_img_feats']
            with gzip.open(img_feats_path, 'r') as f:
                self.img_feats = json.load(f)
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def get_img_feats(self, key):
        feats = []
        for i in range(self.total_views):
            feat = np.array(self.img_feats[f'{key}-{i}'])
            feats.append(feat)
        return np.array(feats)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # get keys
        entry_idx = entry['ans'] if 'ans' in entry else -1 # test set does not contain answers
        if len(entry['objects']) == 2:
            key1, key2 = entry['objects']

        # fix missing key in pair
        else:
            key1 = entry['objects'][entry_idx]
            while True:
                key2 = np.random.choice(list(self.img_feats.keys())).split("-")[0]
                if key2 != key1:
                    break

        # annotation
        annotation = entry['annotation']
        is_visual = entry['visual'] if 'ans' in entry else -1 # test set does not have labels for visual and non-visual categories

        # feats
        start_idx = 6 # discard first 6 views that are top and bottom viewpoints
        img1_n_feats = torch.from_numpy(self.get_img_feats(key1))[start_idx:]
        img2_n_feats = torch.from_numpy(self.get_img_feats(key2))[start_idx:]
        lang_feats = torch.from_numpy(np.array(self.lang_feats[annotation]))


        # label
        ans = entry_idx

        return (
            (img1_n_feats, img2_n_feats),
            lang_feats,
            ans,
            (key1, key2),
            annotation,
            is_visual,
        )