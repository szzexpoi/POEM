import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from utils.misc import convert_to_one_hot
from IPython import embed

MODULE_INPUT_NUM = {
    '_NoOp': 1,
    '_Find': 0,
    '_Relate': 1,
    '_Filter': 1,
    '_And': 2,
    '_Describe': 1,

}

MODULE_OUTPUT_NUM = {
    '_NoOp': 0,
    '_Find': 1,
    '_Relate': 1,
    '_Filter': 1,
    '_And': 1,
    '_Describe': 1,

}

"""
att_stack: (batch_size, dim_att, glimpse, stack_len)
att: (batch_size, dim_att, glimpse)
stack_ptr: (batch_size, stack_len)
mem: (batch_size, dim_vision * glimpse)
"""

class NoOpModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim_hidden = kwargs['dim_hidden']

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        return att_stack, stack_ptr, mem_in, torch.zeros(len(att_stack),self.dim_hidden).cuda() 


class AndModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim_hidden = kwargs['dim_hidden']

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        att2 = _read_from_stack(att_stack, stack_ptr)
        att_stack = _write_to_stack(att_stack, stack_ptr, torch.zeros(feat.size(0),feat.size(1),1).to(feat.device))
        stack_ptr = _move_ptr_bw(stack_ptr)
        att1 = _read_from_stack(att_stack, stack_ptr)
        att_out = torch.min(att1, att2)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
        return att_stack, stack_ptr, mem_in.clone().zero_(), torch.zeros(len(att_stack),self.dim_hidden).cuda() 


class FindModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        concept_weight = torch.load('./prototype_vqa.pt') # glove prototypes
        self.num_concept, self.concept_dim = concept_weight.shape 
        self.mem2concept = nn.Linear(kwargs['dim_v'],self.concept_dim) # dim_vision x dim
        self.prototype_layer = nn.Linear(self.concept_dim,self.num_concept, bias=False)
        self.prototype_layer.weight.data = concept_weight
        for para in self.prototype_layer.parameters():
            para.requires_grad = False # originally False

        self.map_c = nn.Linear(kwargs['dim_hidden'], self.num_concept)
        self.x_conv = nn.Linear(self.num_concept, kwargs['glimpses'])
        self.drop = nn.Dropout(kwargs['dropout_prob'])
        self.fusion = Fusion()

        self.sem2hidden = nn.Linear(self.num_concept*kwargs['glimpses'],kwargs['dim_hidden'])


    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        query = torch.tanh(self.map_c(self.drop(c_i))).unsqueeze(1) # (batch_size, 1, dim_v)

        # prototype
        mem_concept = self.mem2concept(feat)
        feat = torch.tanh(self.prototype_layer(mem_concept)) # (batch_size, num_feat, num_concept)

        x = self.fusion(feat, query)
        att_out = self.x_conv(self.drop(x)) # (batch_size, num_feat, glimpse)
        att_out = F.softmax(att_out, dim=1)  # (batch_size, num_feat, glimpse)
        #att_out = torch.sigmoid(att_out)
        stack_ptr = _move_ptr_fw(stack_ptr)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)


        att_sem = torch.bmm(att_out.transpose(2,1),feat).view(len(feat),-1)
        semantic = self.sem2hidden(att_sem)

        return att_stack, stack_ptr, mem_in.clone().zero_(), semantic

class AttrModule(nn.Module):
    # same as find, but localizing attribute instead of objects
    def __init__(self, **kwargs):
        super().__init__()
        concept_weight = torch.load('./prototype_vqa.pt') # glove prototypes
        self.num_concept, self.concept_dim = concept_weight.shape 
        self.mem2concept = nn.Linear(kwargs['dim_v'],self.concept_dim) # dim_vision x dim
        self.prototype_layer = nn.Linear(self.concept_dim,self.num_concept, bias=False)
        self.prototype_layer.weight.data = concept_weight
        for para in self.prototype_layer.parameters():
            para.requires_grad = False # originally False

        self.map_c = nn.Linear(kwargs['dim_hidden'], self.num_concept)
        self.x_conv = nn.Linear(self.num_concept, kwargs['glimpses'])
        self.drop = nn.Dropout(kwargs['dropout_prob'])
        self.fusion = Fusion()


    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        query = torch.tanh(self.map_c(self.drop(c_i))).unsqueeze(1) # (batch_size, 1, dim_v)

        # prototype
        mem_concept = self.mem2concept(feat)
        feat = torch.tanh(self.prototype_layer(mem_concept))

        x = self.fusion(feat, query)
        att_out = self.x_conv(self.drop(x)) # (batch_size, num_feat, glimpse)
        att_out = F.softmax(att_out, dim=1)  # (batch_size, num_feat, glimpse)
        #att_out = torch.sigmoid(att_out)
        stack_ptr = _move_ptr_fw(stack_ptr)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
        return att_stack, stack_ptr, mem_in.clone().zero_(), feat

class FilterModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.Find = AttrModule(**kwargs)
        self.And = AndModule(**kwargs)

        concept_weight = torch.load('./prototype_vqa.pt') # glove prototypes

        self.num_concept, self.concept_dim = concept_weight.shape 
        self.sem2hidden = nn.Linear(self.num_concept*kwargs['glimpses'],kwargs['dim_hidden'])

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        att_stack, stack_ptr, _, feat = self.Find(vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in)
        att_stack, stack_ptr, _, _ = self.And(vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in)
        att_out = _read_from_stack(att_stack, stack_ptr)
        att_sem = torch.bmm(att_out.transpose(2,1),feat).view(len(feat),-1)
        semantic = self.sem2hidden(att_sem)
       
        return att_stack, stack_ptr, mem_in.clone().zero_(), semantic

class RelateModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.map_c = nn.Linear(kwargs['dim_hidden'], kwargs['dim_edge'])
        self.map_weight = nn.Linear(kwargs['dim_edge'], 1)
        self.glimpses = kwargs['glimpses']
        self.p_length = kwargs['p_length']
        self.p_decay = kwargs['p_decay']
        self.dim_hidden = kwargs['dim_hidden']

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        batch_size = feat_edge.size(0)
        num_feat = relation_mask.size(1)
        query = self.map_c(c_i).view(batch_size, 1, 1, -1).expand_as(feat_edge)
        elt_prod = query * feat_edge
        weit_matrix = F.sigmoid(torch.sum(elt_prod, dim=3))
        
        relation_masks = torch.zeros([self.p_length, batch_size, num_feat, num_feat]).to(stack_ptr.device)
        relation_masks[0] = relation_mask
        for i in range(1, self.p_length):
            relation_masks[i] = torch.matmul(relation_masks[i-1], relation_mask.float())*self.p_decay
        final_mask = torch.sum(relation_masks, axis=0)
        weit_matrix = weit_matrix * final_mask.float()

        att_in = _read_from_stack(att_stack, stack_ptr).permute(0,2,1) #(batch_size, glimpse, att_dim)
        att_out = torch.matmul(att_in, weit_matrix).permute(0,2,1) #(batch_size, att_dim, glimpse)
        norm = torch.max(att_out, dim=1, keepdim=True)[0].detach()
        norm[norm<=1] = 1
        att_out /= norm
        # ---------
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
        return att_stack, stack_ptr, mem_in.clone().zero_(), torch.zeros(len(att_stack),self.dim_hidden).cuda() 


class DescribeModule(nn.Module):
    # glove prototypes
    def __init__(self, **kwargs):
        super().__init__()
        concept_weight = torch.load('./prototype_vqa.pt') # glove prototypes

        self.num_concept, self.concept_dim = concept_weight.shape
        self.map_c = nn.Linear(kwargs['dim_hidden'], 2048) 
        self.mem2concept = nn.Linear(2048,self.concept_dim) # dim_vision x dim
        self.prototype_layer = nn.Linear(self.concept_dim,self.num_concept, bias=False)
        self.prototype_layer.weight.data = concept_weight
        for para in self.prototype_layer.parameters():
            para.requires_grad = False
        self.sem2hidden = nn.Linear(self.num_concept*kwargs['glimpses'],kwargs['dim_hidden'])
        self.drop = nn.Dropout(kwargs['dropout_prob'])

    def forward(self, vision_feat, feat, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem_in):
        batch_size = feat.size(0)
        att_in = _read_from_stack(att_stack, stack_ptr).permute(0,2,1)
        mem_out = torch.bmm(att_in, vision_feat)
        query = torch.tanh(self.map_c(self.drop(c_i))).unsqueeze(1) # (batch_size, 1, dim_v)

        # prototype
        mem_concept = self.mem2concept(mem_out*query)
        mem_concept = torch.tanh(self.prototype_layer(mem_concept)).view(batch_size, -1)
        semantic = self.sem2hidden(mem_concept)
        mem_out = mem_out.view(batch_size, -1) #(batch_size, glimpse*dim_vision)

        return att_stack, stack_ptr, mem_out, semantic


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return - (x - y)**2 + F.relu(x + y)



def _move_ptr_fw(stack_ptr):
    """
    Move the stack pointer forward (i.e. to push to stack).
    stack_ptr: (batch_size, stack_len)
    Return: (batch_size, stack_len)
    """
    filter_fw = torch.FloatTensor([1, 0, 0]).view(1, 1, 3).to(stack_ptr.device)
    batch_size, stack_len = stack_ptr.size()
    new_stack_ptr = F.conv1d(stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)
    # when the stack pointer is already at the stack top, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    stack_top_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_top_mask[stack_len - 1] = 1 # [stack_len, ]
    new_stack_ptr += stack_top_mask * stack_ptr
    return new_stack_ptr


def _move_ptr_bw(stack_ptr):
    """
    Move the stack pointer backward (i.e. to pop from stack).
    """
    filter_fw = torch.FloatTensor([0, 0, 1]).view(1, 1, 3).to(stack_ptr.device)
    batch_size, stack_len = stack_ptr.size()
    new_stack_ptr = F.conv1d(stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)
    # when the stack pointer is already at the stack bottom, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    stack_bottom_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_bottom_mask[0] = 1
    new_stack_ptr += stack_bottom_mask * stack_ptr
    return new_stack_ptr


def _read_from_stack(att_stack, stack_ptr):
    """
    Read the value at the given stack pointer.
    """
    batch_size, stack_len = stack_ptr.size()
    stack_ptr_expand = stack_ptr.view(batch_size, 1, 1, stack_len)
    # The stack pointer is a one-hot vector, so just do dot product
    att = torch.sum(att_stack * stack_ptr_expand, dim=-1)
    return att # (batch_size, att_dim, glimpse)


def _write_to_stack(att_stack, stack_ptr, att):
    """
    Write value 'att' into the stack at the given stack pointer. Note that the
    result needs to be assigned back to att_stack
    """
    batch_size, stack_len = stack_ptr.size()
    stack_ptr_expand = stack_ptr.view(batch_size, 1, 1, stack_len)
    if att.dim() == 3:
        att = att.unsqueeze(3)
    att_stack = att * stack_ptr_expand + att_stack * (1 - stack_ptr_expand)
    return att_stack # (batch_size, att_dim, glimpse, stack_len)


def _sharpen_ptr(stack_ptr, hard):
    """
    Sharpen the stack pointers into (nearly) one-hot vectors, using argmax
    or softmax. The stack values should always sum up to one for each instance.
    """
    if hard:
        # hard (non-differentiable) sharpening with argmax
        stack_len = stack_ptr.size(1)
        new_stack_ptr_indices = torch.argmax(stack_ptr, dim=1)[1]
        new_stack_ptr = convert_to_one_hot(new_stack_ptr_indices, stack_len)
    else:
        # soft (differentiable) sharpening with softmax
        temperature = 0.1
        new_stack_ptr = F.softmax(stack_ptr / temperature, dim=1)
    return new_stack_ptr


def _build_module_validity_mat(stack_len, module_names):
    """
    Build a module validity matrix, ensuring that only valid modules will have
    non-zero probabilities. A module is only valid to run if there are enough
    attentions to be popped from the stack, and have space to push into
    (e.g. _Find), so that stack will not underflow or overflow by design.

    module_validity_mat is a stack_len x num_module matrix, and is used to
    multiply with stack_ptr to get validity boolean vector for the modules.
    """

    module_validity_mat = np.zeros((stack_len, len(module_names)), np.float32)
    for n_m, m in enumerate(module_names):
        # a module can be run only when stack ptr position satisfies
        # (min_ptr_pos <= ptr <= max_ptr_pos), where max_ptr_pos is inclusive
        # 1) minimum position:
        #    stack need to have MODULE_INPUT_NUM[m] things to pop from
        min_ptr_pos = MODULE_INPUT_NUM[m]
        # the stack ptr diff=(MODULE_OUTPUT_NUM[m] - MODULE_INPUT_NUM[m])
        # ensure that ptr + diff <= stack_len - 1 (stack top)
        max_ptr_pos = (
            stack_len - 1 + MODULE_INPUT_NUM[m] - MODULE_OUTPUT_NUM[m])
        module_validity_mat[min_ptr_pos:max_ptr_pos+1, n_m] = 1.

    return module_validity_mat
