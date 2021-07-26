# SNARE Dataset

SNARE dataset and code for MATCH and LaGOR models. 

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
Download pre-extracted image features, language features, and pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1rExJT7LYJ0piZz6s54PaLOKWNElbuGrU?usp=sharing) and put them in the `data/` folder. 

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
