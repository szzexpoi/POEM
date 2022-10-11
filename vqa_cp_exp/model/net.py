import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from . import proto_modules as modules # use prototypical modules instead of standard neural modules (best)

from .questionEncoder import BiGRUEncoder
from .controller import Controller

class POEM_LMH(nn.Module):
    def __init__(self, **kwargs):
        """
        kwargs:
             vocab,
             dim_v, # vertex and edge embedding of scene graph
             dim_word, # word embedding
             dim_hidden, # hidden of seq2seq
             dim_vision,
             dim_edge,
             glimpses,
             cls_fc_dim,
             dropout_prob,
             T_ctrl,
             stack_len,
             device,
             use_gumbel,
             use_validity,
        """
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.num_classes = len(self.vocab['answer_token_to_idx'])


        # semantic memory module
        # soft attention
        self.semantic_att = nn.Linear(self.dim_hidden,self.T_ctrl) # assume 5 steps

        self.classifier = Classifier_lmh(
            in_features=(self.glimpses * self.dim_vision+self.dim_hidden, self.dim_hidden),
            mid_features=self.cls_fc_dim,
            drop=self.dropout_prob
            )
        self.bias_lin = torch.nn.Linear(self.cls_fc_dim, 1)
        self.base_classifier = nn.Linear(self.cls_fc_dim, self.num_classes)

        self.map_vision_to_v = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.dim_vision, self.dim_v, bias=False),
                )
        self.map_two_v_to_edge = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.dim_v * 2, self.dim_edge, bias=False),
                )
        self.num_token = len(self.vocab['question_token_to_idx'])
        self.token_embedding = nn.Embedding(self.num_token, self.dim_word)
        self.dropout = nn.Dropout(self.dropout_prob)

        # modules
        self.module_names = modules.MODULE_INPUT_NUM.keys()
        self.num_module = len(self.module_names)
        self.module_funcs = [getattr(modules, m[1:]+'Module')(**kwargs) for m in self.module_names]
        self.module_validity_mat = modules._build_module_validity_mat(self.stack_len, self.module_names)
        self.module_validity_mat = nn.parameter.Parameter(torch.Tensor(self.module_validity_mat),requires_grad=False) # originally True
        self.module_funcs = nn.ModuleList(self.module_funcs)

        # question encoder
        self.question_encoder = BiGRUEncoder(self.dim_word, self.dim_hidden)
        # controller
        controller_kwargs = {
            'num_module': len(self.module_names),
            'dim_lstm': self.dim_hidden,
            'T_ctrl': self.T_ctrl,
            'use_gumbel': self.use_gumbel,
        }
        self.controller = Controller(**controller_kwargs)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.normal_(self.token_embedding.weight, mean=0, std=1/np.sqrt(self.dim_word))

        # for LMH
        self.smooth_param = torch.nn.Parameter(
          torch.from_numpy(np.full((1,), -1, dtype=np.float32)))

    def forward(self, questions, questions_len, vision_feat, relation_mask, bias=None):
        """
        Args:
            questions [Tensor] (batch_size, seq_len)
            questions_len [Tensor] (batch_size)
            vision_feat (batch_size, dim_vision, num_feat)
            relation_mask (batch_size, num_feat, num_feat)
        """
        batch_size = len(questions)
        questions = questions.permute(1, 0) # (seq_len, batch_size)
        questions_embedding = self.token_embedding(questions) # (seq_len, batch_size, dim_word)
        questions_embedding = torch.tanh(self.dropout(questions_embedding))
        questions_outputs, questions_hidden = self.question_encoder(questions, questions_embedding, questions_len)
        module_logits, module_probs, c_list, cv_list = self.controller(
            questions_outputs, questions_hidden, questions_embedding, questions_len)
        ## feature processing
        vision_feat = vision_feat / (vision_feat.norm(p=2, dim=1, keepdim=True) + 1e-12)
        feat_inputs = vision_feat.permute(0,2,1) # for VQA-only

        if self.dim_v != self.dim_vision:
            feat_inputs = self.map_vision_to_v(feat_inputs) # (batch_size, num_feat, dim_v)
        num_feat = feat_inputs.size(1)
        feat_inputs_expand_0 = feat_inputs.unsqueeze(1).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_inputs_expand_1 = feat_inputs.unsqueeze(2).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_edge = torch.cat([feat_inputs_expand_0, feat_inputs_expand_1], dim=3) # (bs, num_feat, num_feat, 2*dim_v)
        feat_edge = self.map_two_v_to_edge(feat_edge)

        ## stack initialization
        att_stack = torch.zeros(batch_size, num_feat, self.glimpses, self.stack_len).cuda()
        stack_ptr = torch.zeros(batch_size, self.stack_len).cuda()

        stack_ptr[:, 0] = 1
        mem = torch.zeros(batch_size, self.glimpses * self.dim_vision).cuda()

        ## cache for visualization
        cache_module_prob = []
        cache_att = []
        cache_semantic = []
        cache_mask = []

        for t in range(self.T_ctrl):
            c_i = c_list[t] #(batch_size, dim_hidden)
            module_logit = module_logits[t] # (batch_size, num_module)
            if self.use_validity:
                ##########Original##############
                if t < self.T_ctrl-1:
                    module_validity = torch.matmul(stack_ptr, self.module_validity_mat)
                else: # last step must describe (or proto)
                    # module_validity = torch.zeros(batch_size, self.num_module).to(self.device)
                    module_validity = torch.zeros(batch_size, self.num_module).cuda()
                    module_validity[:, 5] = 1

                module_invalidity = (1 - torch.round(module_validity)).byte() # hard validate
                module_logit.masked_fill_(module_invalidity.bool(), -float('inf'))
                module_prob = F.gumbel_softmax(module_logit, hard=self.use_gumbel)

            else:
                module_prob = module_probs[t]
            module_prob = module_prob.permute(1,0) # (num_module, batch_size)


            # run all modules
            res = [f(vision_feat.permute(0,2,1), feat_inputs, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem) for f in self.module_funcs]
            att_stack_avg = torch.sum(module_prob.view(self.num_module,batch_size,1,1,1) * torch.stack([r[0] for r in res]), dim=0)
            stack_ptr_avg = torch.sum(module_prob.view(self.num_module,batch_size,1) * torch.stack([r[1] for r in res]), dim=0)
            stack_ptr_avg = modules._sharpen_ptr(stack_ptr_avg, hard=False)
            mem_avg = torch.sum(module_prob.view(self.num_module,batch_size,1) * torch.stack([r[2] for r in res]), dim=0)
            semantic_avg = torch.sum(module_prob.view(self.num_module,batch_size,1) * torch.stack([r[-1] for r in res]), dim=0)

            att_stack, stack_ptr, mem = att_stack_avg, stack_ptr_avg, mem_avg
            # cache for visualization
            cache_module_prob.append(module_prob)
            atts = []
            for r in res:
                att = modules._read_from_stack(r[0], r[1]) # (batch_size, att_dim, glimpse)
                atts.append(att)
            cache_att.append(atts)
            cache_semantic.append(semantic_avg)
            cache_mask.append((torch.argmax(module_prob,dim=0)==0))


        ## Part 1. features from scene graph module network. (batch, dim_v)
        ## Part 2. question prior. (batch, dim_hidden)

        # leveraging cached semantic for classification
        cache_semantic = torch.cat([cur.unsqueeze(1) for cur in cache_semantic], dim=1)
        # standard soft attention
        semantic_att = F.softmax(self.semantic_att(questions_hidden),dim=-1).unsqueeze(1)
        cache_semantic = torch.bmm(semantic_att,cache_semantic).squeeze(1)
        mem = torch.cat((mem,cache_semantic),dim=-1)
        cls_feat = self.classifier(mem, questions_hidden)
        others = {
            'module_prob': cache_module_prob, # (T, num_module, batch_size)
            'att': cache_att, # (T, num_module, batch_size, att_dim, glimpse)
            'cv': cv_list, # (T, batch_size, len)
        }

        # modules for LMH
        base_logits = torch.log(F.softmax(self.base_classifier(cls_feat),dim=-1)+1e-16)
        if bias is not None:
            bias_factor = self.bias_lin(cls_feat)
            bias_factor = F.softplus(bias_factor)
            soften_factor = F.sigmoid(self.smooth_param)
            bias = bias + soften_factor.unsqueeze(1)
            bias_logits = bias_factor*torch.log(bias+1e-16)
            predicted_logits = F.softmax(base_logits+bias_logits,dim=-1)
            # compute the entropy of biased logits
            bias_logits = F.softmax(bias_logits,dim=-1)
            entropy = -(bias_logits*torch.log(bias_logits+1e-16)).sum(-1)
            return predicted_logits, entropy, others
        else:
            predicted_logits = base_logits
            return predicted_logits, others

        return predicted_logits, others

"""
The Classifier and Attention is according to the codes of Learning to Count
[https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/model.py]
"""
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return - (x - y)**2 + F.relu(x + y)

class Classifier_lmh(nn.Sequential):
    def __init__(self, in_features, mid_features, drop):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = Fusion()
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.bn = nn.BatchNorm1d(mid_features)

    def forward(self, x, y):
        x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        x = self.drop(self.bn(x))
        return x
