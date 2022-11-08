# Divide and Conquer: Answering Questions with Object Factorization and Compositional Reasoning

This repository implements the PrOtotypical nEural Module network. It contains four key components, stored in the following directories:
- vqa_exp: zero-shot VQA experiments on the VQA v2 and Novel-VQA datasets
- gqa_exp: zero-shot VQA experiments on the GQA dataset,
- vqa_cp_exp: Out-of-distribution VQA experiments on VQA-CP dataset,
- proto_learning: the code for our prototype learning method with object factorization, and
- data: code for data preprocessing

Please refer to the README in each directory for details.

### Disclaimer
We adopt the official implementation of the [XNM](https://github.com/shijx12/XNM-Net) as the backbone model for prototypical reasoning. We use the bottom-up features provided in the following repos: [for VQA](https://github.com/peteanderson80/bottom-up-attention) and [for GQA](https://github.com/airsplay/lxmert). Please refer to these links for further README information.


### Reference
If you use our code or data, please cite our paper:
```
TBD
```
