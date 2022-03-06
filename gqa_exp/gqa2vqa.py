import json
import os
import argparse

parser = argparse.ArgumentParser(description='Converting GQA style annotation to VQA style annotations')
parser.add_argument('--input_dir',type=str, default=None, help='Directory to zero-shot GQA annotation')
parser.add_argument('--save_dir',type=str, default=None, help='Directory for saving data')
args = parser.parse_args()


for split in ['train','val','val_known']:
	data = json.load(open(os.path.join(agrs.input_dir,'novel_gqa_'+split+'.json')))

	processed_question = []
	anno = []

	for qid in data:
		img_id = data[qid]['imageId']
		question = data[qid]['question']
		answer = data[qid]['answer']

		# VQA style question annotations
		cur_que = dict()
		cur_que['question_id'] = qid
		cur_que['question'] = question
		cur_que['image_id'] = img_id
		processed_question.append(cur_que)

		# VQA style answer annotations
		cur_anno = dict()
		cur_anno['answers'] = [{'answer':answer}]
		cur_anno['question_type'] = data[qid]['types']['detailed']
		cur_anno['question_id'] = qid
		cur_anno['image_id'] = img_id
		anno.append(cur_anno)


	with open(os.path.join(args.save_dir,'novel_gqa_'+split+'_question.json'),'w') as f:
		json.dump(processed_question,f)
	with open(os.path.join(args.save_dir,'novel_gqa_'+split+'_annotation.json'),'w') as f:
		json.dump(anno,f)
