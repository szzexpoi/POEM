import pickle
import json
import numpy as np
import os


# reading glove data
glove_embed = dict()
with open('./glove.840B.300d.txt') as f:
	entries = f.readlines()
for entry in entries:
	val = entry.split(' ')
	word = val[0]
	val = [float(cur) for cur in val[1:]]
	glove_embed[word] = np.array(val)

if not os.path.exists(os.path.join('./','data')):
	os.mkdir(os.path.join('./','data'))

with open('./data/glove_embedding.pickle','wb') as f:
	pickle.dump(glove_embed,f)
