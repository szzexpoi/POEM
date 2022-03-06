import json
import os 

data_dir = './GQA' 
save_dir = './VQA'


# for split in ['train','val']:
for split in ['val']:
	# data = json.load(open(os.path.join(data_dir,split+'_balanced_questions.json')))
	# data = json.load(open(os.path.join(data_dir,'ood_'+split+'_tail.json')))
	data = json.load(open(os.path.join(data_dir,'novel_gqa_'+split+'_img_known_latest.json')))

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


	with open(os.path.join(save_dir,'Novel_img_'+split+'_vqa_question_known_latest.json'),'w') as f:
		json.dump(processed_question,f)
	with open(os.path.join(save_dir,'Novel_img_'+split+'_vqa_annotation_known_latest.json'),'w') as f:
		json.dump(anno,f)

	# with open(os.path.join(save_dir,'GQA_'+split+'_vqa_question.json'),'w') as f:
	# 	json.dump(processed_question,f)
	# with open(os.path.join(save_dir,'GQA_'+split+'_vqa_annotation.json'),'w') as f:
	# 	json.dump(anno,f)