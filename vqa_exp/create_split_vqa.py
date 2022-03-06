import json
import numpy as np
import os
from copy import deepcopy
import operator
import argparse

parser = argparse.ArgumentParser(description='Creating zero-shot VQA splits for VQA dataset')
parser.add_argument('--coco_dir',type=str, default=None, help='Directory to mscoco annotations files')
parser.add_argument('--vqa_dir',type=str, default=None, help='Directory to VQA annotations files')
parser.add_argument('--save_dir',type=str, default=None, help='Directory for saving data')
args = parser.parse_args()

train_anno = json.load(open(os.path.join(args.coco_dir,'instances_train2014.json')))
val_anno = json.load(open(os.path.join(args.coco_dir,'instances_val2014.json')))
idx2obj = dict()
for i in range(len(train_anno['categories'])):
    idx2obj[train_anno['categories'][i]['id']] = train_anno['categories'][i]['name']

# re-organize the data
processed_train = dict()
processed_val = dict()
obj_count = dict()
for split_id, cur_data in enumerate([train_anno,val_anno]):
    for i in range(len(cur_data['annotations'])):
        cur_anno = cur_data['annotations'][i]
        cur_bbox = cur_anno['bbox']
        cur_obj = idx2obj[cur_anno['category_id']]
        cur_img = cur_anno['image_id']
        if split_id ==0:
            if cur_img not in processed_train:
                processed_train[cur_img] = dict()
            if cur_obj not in processed_train[cur_img]:
                processed_train[cur_img][cur_obj] = []
            processed_train[cur_img][cur_obj].append(cur_bbox)
        else:
            if cur_img not in processed_val:
                processed_val[cur_img] = dict()
            if cur_obj not in processed_val[cur_img]:
                processed_val[cur_img][cur_obj] = []
            processed_val[cur_img][cur_obj].append(cur_bbox)
        if cur_obj not in obj_count:
            obj_count[cur_obj] = 0
        obj_count[cur_obj]+=1

# select novel concepts
count = sorted(obj_count.items(), key=operator.itemgetter(1))
count.reverse()
selected_concept = np.random.choice([cur[0] for cur in count[-50:]],10,replace=False)
with open('concept_pool.json','w') as f:
    json.dump(list(selected_concept),f)

# filtering images based on the selected concepts
selected_train = dict()
selected_val = dict()
for img in processed_train:
    flag = True
    for cur in selected_concept:
        if cur in processed_train[img]:
            flag = False
            break
    if flag:
        selected_train[img] = 1
for img in processed_val:
    flag = False
    for cur in selected_concept:
        if cur in processed_val[img]:
            flag = True
            break
    if flag:
        selected_val[img] = 1

# read VQA data
vqa_train_question = json.load(open(os.path.join(args.vqa_dir,'v2_OpenEnded_mscoco_train2014_questions.json')))['questions']
vqa_train_anno = json.load(open(os.path.join(args.vqa_dir,'v2_mscoco_train2014_annotations.json')))['annotations']
vqa_val_question = json.load(open(os.path.join(args.vqa_dir,'v2_OpenEnded_mscoco_val2014_questions.json')))['questions']
vqa_val_anno = json.load(open(os.path.join(args.vqa_dir,'v2_mscoco_val2014_annotations.json')))['annotations']

# selecting training questions based on the filtered images
final_train_question = []
final_train_anno = []
for i in range(len(vqa_train_question)):
    if vqa_train_question[i]['image_id'] in selected_train:
        final_train_question.append(vqa_train_question[i])
for i in range(len(vqa_train_anno)):
    if vqa_train_anno[i]['image_id'] in selected_train:
        final_train_anno.append(vqa_train_anno[i])

# selecting validation questions (novel split) based on the filtered images
final_val_question = []
final_val_anno = []
selected_qid = dict()
for i in range(len(vqa_val_question)):
    if vqa_val_question[i]['image_id'] in selected_val:
        final_val_question.append(vqa_val_question[i])
        selected_qid[vqa_val_question[i]['question_id']] = 1

invalid_id = dict()
for i in range(len(vqa_val_anno)):
    if vqa_val_anno[i]['question_id'] in selected_qid:
        if vqa_val_anno[i]['multiple_choice_answer'] not in selected_concept:
            final_val_anno.append(vqa_val_anno[i])
        else:
            invalid_id[vqa_val_anno[i]['question_id']] = 1

# remove question whose answers are the unseen concepts
final_val_question_latest = []
for i in range(len(final_val_question)):
    if final_val_question[i]['question_id'] not in invalid_id:
        final_val_question_latest.append(final_val_question[i])
final_val_question = final_val_question_latest

# create the known validation set
novel_id = dict()
for i in range(len(final_val_question)):
    novel_id[final_val_question[i]['question_id']] = 1

known_question = []
known_anno = []
invalid_id = dict()

for i in range(len(vqa_val_anno)):
    if vqa_val_anno[i]['question_id'] not in novel_id:
        if vqa_val_anno[i]['multiple_choice_answer'] in selected_concept:
            invalid_id[vqa_val_anno[i]['question_id']] = 1
        else:
            known_anno.append(vqa_val_anno[i])
for i in range(len(vqa_val_question)):
    if vqa_val_question[i]['question_id'] not in novel_id and vqa_val_question[i]['question_id'] not in invalid_id:
        known_question.append(vqa_val_question[i])


# save data
with open(os.path.join(args.save_dir,'novel_vqa_train_questions.json'),'w') as f:
    json.dump(final_train_question,f)
with open(os.path.join(args.save_dir,'novel_vqa_train_annotations.json'),'w') as f:
    json.dump(final_train_anno,f)
with open(os.path.join(args.save_dir,'novel_vqa_val_questions.json'),'w') as f:
    json.dump(final_val_question,f)
with open(os.path.join(args.save_dir,'novel_vqa_val_annotations.json'),'w') as f:
    json.dump(final_val_anno,f)
with open(os.path.join(args.save_dir,'novel_vqa_val_questions_known.json'),'w') as f:
    json.dump(known_question,f)
with open(os.path.join(args.save_dir,'novel_vqa_val_annotations_known.json'),'w') as f:
    json.dump(known_anno,f)
