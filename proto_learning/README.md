# Learning Discriminative Prototypes with Object Factorization

This code implements the prototype learning method proposed in POEM.

### Requirements
1. Requirements for Pytorch. We use Pytorch 1.2.0 in our experiments.
2. Requirements for Tensorflow. We only use the tensorboard for visualization.
3. You may need to install the OpenCV package (CV2) for Python.

### Pretrained prototypes
We provide the [pretrained prototypes](https://drive.google.com/file/d/1alCvI8tub2yv0lJI2shjTlC8fA_TEZLe/view?usp=sharing) used in our experiments.

### Training prototype from scratch
If you would like to train the prototype from scratch:
1. Download our [processed annotations](https://drive.google.com/file/d/1t3ZyRKNLL0Pg3hmrOJIUWOjQpv78ahuP/view?usp=sharing) for multi-label classification. Currently we only provide those for datasets in our experiments, code for generating the annotations for other datasets will be released shortly.
2. Training the model:
```
python ./main.py --mode $DATASET --img_dir $FEAT_DIR --checkpoint_dir $CKPT
```
Replace **DATASET** with `vqa`, `gqa`, or `novelvqa` for learning prototypes on the corresponding datasets.
3. Extract the prototypes from the checkpoints. The prototypes are stored as model's weights, open a python terminal:
```
>>> import torch
>>> proto = torch.load('./ckpt/model_best.pt')['prototype.weight']
>>> torch.save(proto,'prototype.pt')
```
