import json
import numpy as np
import os
from copy import deepcopy
import operator
import argparse

parser = argparse.ArgumentParser(description='Creating zero-shot VQA splits for GQA dataset')
parser.add_argument('--question_dir',type=str, default=None, help='Directory to question files')
parser.add_argument('--scene_graph_dir',type=str, default=None, help='Directory to scene graph files')
parser.add_argument('--semantic_dir',type=str, default=None, help='Directory to simplified semantics')
parser.add_argument('--save_dir',type=str, default=None, help='Directory for saving data')
args = parser.parse_args()

train_question = json.load(open(os.path.join(args.question_dir,'train_balanced_questions.json')))
val_question = json.load(open(os.path.join(args.question_dir,'val_balanced_questions.json')))
question = deepcopy(train_question).update(val_question)

scene_graph = json.load(open(os.path.join(args.scene_graph_dir,'train_sceneGraphs.json')))
val_scene_graph = json.load(open(os.path.join(args.scene_graph_dir,'val_sceneGraphs.json')))
scene_graph.update(val_scene_graph)

semantic = json.load(open(os.path.join(args.semantic_dir,'simplified_semantics_train.json')))
val_semantic = json.load(open(os.path.join(args.semantic_dir,'simplified_semantics_val.json')))
semantic.update(val_semantic)

# using object category as concept
concept_pool = dict()
invalid_pool = dict()
for qid in question:
    cur_img_id = question[qid]['imageId']
    cur_semantic = semantic[qid]
    cur_graph = scene_graph[cur_img_id]['objects']
    cur_ans = question[qid]['answer']

    cur_concept = dict()
    for step in cur_semantic:
        for i in range(1,3):
            if step[i] is not None and 'obj:' in step[i]:
                cur_obj = step[i][4:].split(',')[0]
                cur_obj = cur_graph[cur_obj]['name']
                cur_concept[cur_obj] = 1
            elif step[i] is not None and 'obj*:' in step[i]:
                cur_obj = step[i][5:].split(',')[0].replace(' (-)','')
                while cur_obj[-1] == ' ':
                    cur_obj = cur_obj[:-1]
                cur_concept[cur_obj] = 1

    for concept in cur_concept:
        if '(' in concept:
            concept = concept.split(' (')[0]
        if cur_ans != concept:
            if concept not in concept_pool:
                concept_pool[concept] = []
            concept_pool[concept].append(qid)
        else:
            if concept not in invalid_pool:
                invalid_pool[concept] = []
            invalid_pool[concept].append(qid)

# merge plural and singular
merged_concept = dict()
for concept in concept_pool:
    if concept[-1] == 's' and concept[:-1] in concept_pool:
        merged_concept[concept] = concept[:-1]
    elif concept[-2:] == 'es' and concept[:-2] in concept_pool:
        merged_concept[concept] = concept[:-2]

processed_data = deepcopy(concept_pool)
processed_invalid = deepcopy(invalid_pool)
for concept in list(processed_data.keys()):
    if concept in merged_concept:
        processed_data[merged_concept[concept]] = processed_data[merged_concept[concept]] + processed_data[concept]
        del processed_data[concept]

for concept in list(processed_invalid.keys()):
    if concept in merged_concept and concept in processed_invalid:
        if merged_concept[concept] in processed_invalid:
            processed_invalid[merged_concept[concept]] = processed_invalid[merged_concept[concept]] + processed_invalid[concept]
            del processed_invalid[concept]
        else:
            processed_invalid[merged_concept[concept]] = processed_invalid[concept]
            del processed_invalid[concept]

for concept in processed_data:
    processed_data[concept] = set(processed_data[concept])
    if concept in processed_invalid:
        processed_invalid[concept] = set(processed_invalid[concept])


# count the frequency of concepts
count = dict()
for concept in processed_data:
    if concept == 'scene':
        continue
    tmp_dict = dict()
    for qid in processed_data[concept]:
        tmp_dict[question[qid]['imageId']] = 1
    count[concept] = len(tmp_dict)
count = sorted(count.items(), key=operator.itemgetter(1))
count.reverse()

 # randomly sample concepts for removal
selected_concept = np.random.choice([cur[0] for cur in count[:50]],10,replace=False)
with open('concept_pool.json','w') as f:
    json.dump(list(selected_concept),f)

# filtering questions based on the selected concepts for removal
remove_img = dict()
used_img = dict()
for concept in selected_concept:
    for qid in processed_data[concept]:
        if qid in train_question:
            img_id = train_question[qid]['imageId']
            remove_img[img_id] = 1
    if concept in processed_invalid:
        for qid in processed_invalid[concept]:
            if qid in train_question:
                img_id = train_question[qid]['imageId']
                remove_img[img_id] = 1
            elif qid in val_question:
                del val_question[qid]

# remove training questions with the novel concepts
for qid in list(train_question.keys()):
    img_id = train_question[qid]['imageId']
    if img_id in remove_img:
        del train_question[qid]
    else:
        used_img[img_id] = 1

# finalize validation questions (novel split)
final_val = dict()
for concept in selected_concept:
    for qid in processed_data[concept]:
        if qid in val_question:
            val_count += 1
            final_val[qid] = val_question[qid]

# creating known validation split
ori_val = json.load(open(os.path.join(args.question_dir,'val_balanced_questions.json')))
known_split = dict()
for qid in ori_val:
    if qid not in final_val and ori_val[qid]['answer'] not in selected_concept:
        known_split[qid] = ori_val[qid]

# saving data
with open(os.path.join(save_dir,'novel_gqa_train.json'),'w') as f:
    json.dump(processed_train,f)
with open(os.path.join(save_dir,'novel_gqa_val.json'),'w') as f:
    json.dump(final_val,f)
with open(os.path.join(save_dir,'novel_gqa_val_known.json'),'w') as f:
    json.dump(known_split,f)
