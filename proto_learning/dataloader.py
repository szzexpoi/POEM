import numpy as np
import random
import os
import time
import operator
import torch
import torch.utils.data as data
import json
import gc
import pickle
from transformers import BertTokenizer

# convert problematic answer
ANS_CONVERT = {
    "a man": "man",
    "the man": "man",
    "a woman": "woman",
    "the woman": "woman",
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
    'grey': 'gray',
}

class Batch_generator_prototype(data.Dataset):
	def __init__(self,img_dir,data_dir,mode='train'):
		self.img_dir = img_dir
		self.data = json.load(open(os.path.join(data_dir,'prototype_data_gqa.json')))[mode]
		self.obj2idx = json.load(open(os.path.join(data_dir,'obj2idx_gqa.json')))

		self.init_data()

	def init_data(self,):
		self.img = []
		self.label = []

		for img_id in self.data:
			self.img.append(img_id)
			self.label.append(self.data[img_id])

	def __getitem__(self,index):
		img_id = self.img[index]
		label = self.label[index]

		# load image features
		img = np.load(os.path.join(self.img_dir,str(img_id)+'.npy'))

		# standard multi-label classification setting
		mask = torch.zeros(len(self.obj2idx))
		for obj in label:
			mask[self.obj2idx[obj]] = 1

		return img, mask, img_id

	def __len__(self,):
		return len(self.img)


class Batch_generator_prototype_VQA(data.Dataset):
	def __init__(self,img_dir,data_dir,mode='train',dataset='vqa'):
		self.img_dir = img_dir
        if dataset == 'vqa':
    		self.data = json.load(open(os.path.join(data_dir,'prototype_data_vqa.json')))[mode]
    		self.obj2idx = json.load(open(os.path.join(data_dir,'obj2idx_vqa.json')))
        else:
    		self.data = json.load(open(os.path.join(data_dir,'prototype_data_vqa_all.json')))[mode]
    		self.obj2idx = json.load(open(os.path.join(data_dir,'obj2idx_vqa_all.json')))

		self.init_data()

	def init_data(self,):
		self.img = []
		self.label = []

		for img_id in self.data:
			self.img.append(img_id)
			cur_label = []
			for k in self.data[img_id]:
				cur_label.append(self.obj2idx[k])
			self.label.append(cur_label)

	def __getitem__(self,index):
		img_id = self.img[index]
		label = self.label[index]

		# load image features
		img = np.load(os.path.join(self.img_dir,str(img_id)+'.npy'))

		# standard multi-label classification setting
		mask = torch.zeros(len(self.obj2idx))
		for obj in label:
			mask[obj] = 1

		return img, mask, img_id

	def __len__(self,):
		return len(self.img)
