# OOD VQA experiments on the VQA-CP dataset

### Data preparation
Please follow the repo of [XNM](https://github.com/shijx12/XNM-Net/tree/master/exp_vqa) for:
1. Downloading and unzip [Glove Vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip)
2. Process the Glove vectors:
```
python process_glove.py
```
3. Preprocess bottom-up features

### Question generation
1. Download the VQA-CP question and annotation files [here](https://computing.ece.vt.edu/~aish/vqacp/). Note that we use the latest version of the dataset (VQA-CP v2).

2. Generate question files for training
```
python preprocess/preprocess_questions.py --glove_pt ./data/glove_embedding.pickle --input_questions_json vqacp_v2_train_questions.json --input_annotations_json vqacp_v2_train_annotations.json --output_pt ./data/train_questions_vqa_cp.pt --vocab_json ./data/vocab_vqa_cp.json --mode train --val_questions_json vqacp_v2_test_questions.json
```
3. Generate question files for evaluation
```
python preprocess/preprocess_questions.py --glove_pt ./data/glove_embedding.pickle --input_questions_json vqacp_v2_test_questions.json --input_annotations_json vqacp_v2_test_annotations.json --output_pt ./data/test_questions_vqa_cp.pt --vocab_json ./data/vocab_vqa_cp.json --mode val --val_questions_json vqacp_v2_test_questions.json
```

### Training
```
python train.py --input_dir ./data --feature_dir $FEAT_DIR --save_dir $SAVE_DIR --val --train_question_pt train_questions_vqa_cp.pt --val_question_pt test_questions_vqa_cp.pt --vocab_json vocab_vqa.json
```

### Evaluation
For evaluation on the test split:
```
python validate.py --input_dir ./data --feature_dir $FEAT_DIR --ckpt $SAVE_DIR/model.pt --val_question_pt test_questions_vqa_cp.pt --vocab_json vocab_vqa_cp.json
```
