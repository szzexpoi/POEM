import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Prototype_Module(nn.Module):
	def __init__(self,img_size,num_prototype,num_concept):
		super(Prototype_Module,self).__init__()
		self.num_prototype = num_prototype

		# raw features
		self.prototype = nn.Linear(img_size, num_prototype,bias=False)
		self.hidden_size = img_size		
		self.proto2concept = nn.Linear(self.hidden_size,num_concept) 

		self.attention_layer = nn.Linear(self.hidden_size,1)

	def forward(self,img):
		batch = len(img)
		proto_sim = torch.sigmoid(self.prototype(img)) # originally tanh
		merged_proto = torch.bmm(proto_sim,self.prototype.weight.unsqueeze(0).expand(batch,self.num_prototype,self.hidden_size))
		# attentive prediction
		prediction = torch.sigmoid(self.proto2concept(merged_proto))
		att = F.softmax(self.attention_layer(merged_proto),dim=1)
		prediction = (prediction*att).sum(1)

		return prediction
