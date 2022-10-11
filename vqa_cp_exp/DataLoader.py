# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import numpy as np
import json
import pickle
import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from IPython import embed
import os


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    return vocab


def load_property_concept(path):
    with open(path, 'r') as f:
        pc = json.load(f)
    return pc


class QADataset(Dataset):

    def __init__(self, answers, questions, questions_len, q_image_indices,
                       feature_h5, feat_coco_id_to_index, num_answer, use_spatial, question_type, bias, question_id, mode):
        # convert data to tensor
        self.all_answers = answers
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_questions_len = torch.LongTensor(np.asarray(questions_len))
        self.all_question_type = question_type
        self.bias_mapping = bias
        self.all_q_image_idxs = torch.LongTensor(np.asarray(q_image_indices)) # for VQA
        # self.all_q_image_idxs = q_image_indices # for GQA

        self.feature_h5 = feature_h5
        self.feat_coco_id_to_index = feat_coco_id_to_index
        self.num_answer = num_answer
        self.use_spatial = use_spatial
        self.mode = mode
        self.question_id = question_id

        # # for loading npy features
        # self.img_size = json.load(open(os.path.join(feature_h5,'img_size.json')))

        # # for program supervision
        # if self.mode == 'train':
        #     self.program = json.load(open('./data/program.json'))
        #     self.program2idx = {'noop':0,'find':1,'transform':2,'filter':3,'and':4,'describe':5}


        # # relationship for clip features
        # pos_idx = np.zeros([7,7,2])
        # for i in range(7):
        #     for j in range(7):
        #         pos_idx[i,j,0] = i 
        #         pos_idx[i,j,1] = j
        # pos_idx = pos_idx.reshape([49,2])
        # self.clip_relation = torch.zeros(49,49)
        # for i in range(49):
        #     for j in range(i+1,49):
        #         if np.abs(pos_idx[i][0]-pos_idx[j][0])>1 or np.abs(pos_idx[i][1]-pos_idx[j][1])>1:
        #             pass
        #         else:
        #             self.clip_relation[i,j] = self.clip_relation[j,i] = 1


    def __getitem__(self, index):
        answer = self.all_answers[index] if self.all_answers is not None else None
        if answer is not None:
            _answer = torch.zeros(self.num_answer)
            for i in answer:
                _answer[i] += 1
            answer = _answer
        question = self.all_questions[index]
        question_len = self.all_questions_len[index]
        question_type = self.all_question_type[index]
        
        # fetch vision features (h5, VQA)
        image_idx = self.all_q_image_idxs[index].item() # coco_id
        feat_index = self.feat_coco_id_to_index[image_idx]
        with h5py.File(self.feature_h5, 'r') as f:
            vision_feat = f['features'][feat_index]
            boxes = f['boxes'][feat_index]
            w = f['widths'][feat_index]
            h = f['heights'][feat_index]

        # # fetch vision features (unziped npy)
        # image_idx = self.all_q_image_idxs[index] # coco_id
        # vision_feat = np.load(os.path.join(self.feature_h5,'feature',str(image_idx)+'.npy'))
        # boxes = np.load(os.path.join(self.feature_h5,'box',str(image_idx)+'.npy'))
        # boxes = boxes.transpose()
        # vision_feat = vision_feat.transpose()
        # h, w = self.img_size[image_idx]

        # # unziped CLIP features
        # # image_idx = self.all_q_image_idxs[index] # gqa
        # image_idx = self.all_q_image_idxs[index].item() # coco
        # vision_feat = np.load(os.path.join(self.feature_h5,str(image_idx)+'.npy'))
        # vision_feat = vision_feat.transpose()
        # spatial_feat = np.zeros((5, 49))


        # (non-VinVl spatial features)
        spatial_feat = np.zeros((5, len(boxes[0])))
        spatial_feat[0, :] = boxes[0, :] * 2 / w - 1 # x1
        spatial_feat[1, :] = boxes[1, :] * 2 / h - 1 # y1
        spatial_feat[2, :] = boxes[2, :] * 2 / w - 1 # x2
        spatial_feat[3, :] = boxes[3, :] * 2 / h - 1 # y2
        spatial_feat[4, :] = (spatial_feat[2, :]-spatial_feat[0, :]) * (spatial_feat[3, :]-spatial_feat[1, :])

        # # fetch vision features (VinVL, npy)
        # image_idx = self.all_q_image_idxs[index] # coco_id
        # raw_feat = np.load(os.path.join(self.feature_h5,str(image_idx)+'.npy'))
        # valid_num = len(raw_feat)
        # vision_feat = np.zeros([100,2048]).astype('float32') # maximum bbox 100
        # vision_feat[:valid_num] = raw_feat[:,:-6]
        # vision_feat = vision_feat.transpose()
        # bbox_feat = np.zeros([100,6]).astype('float32') # maximum bbox 100
        # bbox_feat[:valid_num] = raw_feat[:,-6:]
        # boxes = bbox_feat[:,:-2]
        # boxes = boxes.transpose()

        # spatial_feat = np.zeros((5, 100))
        # spatial_feat[0, :] = bbox_feat[:,0] # x1
        # spatial_feat[1, :] = bbox_feat[:,1] # y1
        # spatial_feat[2, :] = bbox_feat[:,2] # x2
        # spatial_feat[3, :] = bbox_feat[:,3] # y2
        # spatial_feat[4, :] = (bbox_feat[:,4] * bbox_feat[:,5])*4 # w*h

        if self.use_spatial:
            vision_feat = np.concatenate((vision_feat, spatial_feat), axis=0)
        vision_feat = torch.from_numpy(vision_feat).float()
        #########

        # for standard relationship
        num_feat = boxes.shape[1]
        relation_mask = np.zeros((num_feat, num_feat))
        for i in range(num_feat):
            for j in range(i+1, num_feat):
                # if there is no overlap between two bounding box
                if boxes[0,i]>boxes[2,j] or boxes[0,j]>boxes[2,i] or boxes[1,i]>boxes[3,j] or boxes[1,j]>boxes[3,i]:
                    pass
                else:
                    relation_mask[i,j] = relation_mask[j,i] = 1
        relation_mask = torch.from_numpy(relation_mask).byte()

        # # for clip relationship based on vincinity of regions
        # relation_mask = self.clip_relation.byte()


        # return (image_idx, answer, question, question_len, vision_feat, relation_mask) # standard VQA
        # return (answer, question, question_len, vision_feat, relation_mask) # for GQA
        # return (self.question_id[index], answer, question, question_len, vision_feat, relation_mask) # for GQA-visualization


        # for LMH
        if self.mode == 'train':
            bias = torch.from_numpy(self.bias_mapping[question_type])
            return (image_idx, answer, question, question_len, vision_feat, relation_mask, bias)
            # return (answer, question, question_len, vision_feat, relation_mask, bias) # for GQA
        else:
            return (image_idx, answer, question, question_len, vision_feat, relation_mask)
            # return (answer, question, question_len, vision_feat, relation_mask) # for GQA

        # # for program supervision
        # if self.mode == 'train':
        #     cur_qid = self.question_id[index]
        #     cur_program = self.program[cur_qid]
        #     program_label = torch.zeros(5,6) # 5 steps * 6 types of modules
        #     for idx,step in enumerate(cur_program):
        #         program_label[idx,self.program2idx[step]] = 1
        #     return (program_label,answer, question, question_len, vision_feat, relation_mask)
        # else:
        #     return (answer, question, question_len, vision_feat, relation_mask)


    def __len__(self):
        return len(self.all_questions)



class QADataLoader(DataLoader):

    def __init__(self, **kwargs):
        vocab_json_path = str(kwargs.pop('vocab_json'))
        print('loading vocab from %s' % (vocab_json_path))
        vocab = load_vocab(vocab_json_path)

        question_pt_path = str(kwargs.pop('question_pt'))
        print('loading questions from %s' % (question_pt_path))
        with open(question_pt_path, 'rb') as f:
            obj = pickle.load(f)
            questions = obj['questions']
            questions_len = obj['questions_len']
            q_image_indices = obj['image_idxs']
            answers = obj['answers']
            glove_matrix = obj['glove']
            question_type = obj['question_type'] # for LMH
            bias = obj['bias'] # for LMH
            # question_id = obj['question_id'] # for GQA w/ program supervision
            question_id = [] # for VQA
        
        use_spatial = kwargs.pop('use_spatial')
        mode = kwargs.pop('mode')
      
        # for VQA
        with h5py.File(kwargs['feature_h5'], 'r') as features_file:
            coco_ids = features_file['ids'][()]
        feat_coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
      
        # # for GQA
        # feat_coco_id_to_index = {}


        self.feature_h5 = kwargs.pop('feature_h5')
        self.dataset = QADataset(answers, questions, questions_len, q_image_indices,
                self.feature_h5, feat_coco_id_to_index, len(vocab['answer_token_to_idx']), use_spatial, question_type, bias, question_id, mode)
        
        self.vocab = vocab
        self.batch_size = kwargs['batch_size']
        self.glove_matrix = glove_matrix

        # collect concept (NMN)
        # self.property_concepts = load_property_concept(str(kwargs.pop('pc_json')))
        # self.property = torch.tensor(list(self.property_concept.keys())+['identity']+['relation'])
        # ordered_C = [
        # torch.stack([concept for concept in self.property_concept[property]])
        # for property in self.property]
        # self.concept = torch.cat(ordered_C, dim=0)

        # kwargs['collate_fn'] = default_collate
        # super().__init__(self.dataset, **kwargs)

        # # collect concept (NSM)
        # self.property_concept = load_property_concept(str(kwargs.pop('pc_json')))
        # # self.property = torch.tensor(list(self.property_concept.keys()))
        # self.property = list(self.property_concept.keys())

        # ordered_C = [
        # torch.stack([concept for concept in self.property_concept[property]])
        # for property in self.property]
        # self.concept = torch.cat(ordered_C, dim=0)

        del kwargs['pc_json']

        kwargs['collate_fn'] = default_collate
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)