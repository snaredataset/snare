# SNARE Dataset

SNARE dataset and code for MATCH and LaGOR models. 

## Paper and Citation

[Language Grounding with 3D Objects](https://arxiv.org/abs/2107.12514)

```
@article{snare,
  title={Language Grounding with {3D} Objects},
  author={Jesse Thomason and Mohit Shridhar and Yonatan Bisk and Chris Paxton and Luke Zettlemoyer},
  journal={arXiv},
  year={2021},
  url={https://arxiv.org/abs/2107.12514}
}
```

## Installation

#### Clone
```bash
$ git clone https://github.com/snaredataset/snare.git

$ virtualenv -p $(which python3) --system-site-packages snare_env # or whichever package manager you prefer
$ source snare_env/bin/activate

$ pip install --upgrade pip
$ pip install -r requirements.txt
```  
Edit `root_dir` in [cfgs/train.yaml](cfgs/train.yaml) to reflect your working directory.

#### Download Data and Checkpoints 
Download pre-extracted image features, language features, and pre-trained checkpoints from [here](https://drive.google.com/drive/folders/18sKN1MawcCjqQ4nbe6m4XAcWogWClKGe?usp=share_link) and put them in the `data/` folder. 

## Usage

#### Zero-shot CLIP Classifier
```bash
$ python train.py train.model=zero_shot_cls train.aggregator.type=maxpool 
```

#### MATCH
```bash
$ python train.py train.model=single_cls train.aggregator.type=maxpool 
```

#### LaGOR
```bash
$ python train.py train.model=rotator train.aggregator.type=two_random_index train.lr=5e-5 train.rotator.pretrained_cls=<path_to_pretrained_single_cls_ckpt>
```

## Scripts

Run [`scripts/train_classifiers.sh`](scripts/train_classifiers.sh) and [`scripts/train_rotators.sh`](scripts/train_rotators.sh) to reproduce the results from the paper.

To train the rotators, edit [`scripts/train_rotators.sh`](scripts/train_rotators.sh) and replace the `PRETRAINED_CLS` with the path to the checkpoint you wish to use to train the rotator:
```
PRETRAINED_CLS="<root_path>/clip-single_cls-random_index/checkpoints/<ckpt_name>.ckpt'"
```

## Preprocessing

If you want to extract CLIP vision and language features from raw images:

1. Download models-screenshot.zip from [ShapeNetSem](https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetSem.v0/models-screenshots.zip), and extract it inside `./data/`.
2. Edit and run `python scripts/extract_clip_features.py` to save `shapenet-clipViT32-frames.json.gz` and `langfeat-512-clipViT32.json.gz` 

## Leaderboard

Please send your `...test.json` prediction results to [Mohit Shridhar](mailto:mshr@cs.washington.edu). We will get back to you as soon as possible. 

**Instructions**:
- Include a name for your model, your team name, and affiliation (if not anonymous).
- Submissions are limited to a maximum of one per week. Please do not create fake email accounts and send multiple submissions.  

**Rankings**:

| Rank | Model                       | All  | Visual | Blind |
|------|-----------------------------|------|:------:|:-----:|
| 1    | **Transformer Classifier** <br>(Anonymous)<br>8 Jun 2023  | 81.7 |  87.7  |  75.4 |
| 2    | **VLG** <br>[(Corona et al.)](https://arxiv.org/abs/2205.09710)<br>15 Mar 2022  | 79.0 |  86.0  |  71.7 |
| 2    | **LOCKET** <br>(Anonymous)<br>14 Oct 2022  | 79.0 |  86.1  |  71.5 |
| 4    | **VLG** <br>[(Corona et al.)](https://arxiv.org/abs/2205.09710)<br>13 Nov 2021  | 78.7 |  85.8  |  71.3 |
| 5    | **LOCKET** <br>(Anonymous)<br>23 Oct 2022  | 77.7 |  85.5  |  69.5 |
| 6    | **LAGOR** <br>[(Thomason et. al)](https://arxiv.org/pdf/2107.12514.pdf)<br>15 Sep 2021 | 77.0 |  84.3  |  69.4 |
| 7    | **MATCH** <br>[(Thomason et. al)](https://arxiv.org/pdf/2107.12514.pdf)<br>15 Sep 2021 | 76.4 | 83.7   | 68.7  |
