import torch
epsilon = 1e-15

def bce_loss(input,target):
	loss = -target*torch.log(torch.clamp(input,min=epsilon,max=1)) - (1-target)*torch.log(torch.clamp(1-input,min=epsilon,max=1))
	loss = loss.sum(-1)
	return loss.mean()