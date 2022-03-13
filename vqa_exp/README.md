# Zero-shot VQA experiments on the VQA and Novel-VQA dataset

### Data preparation
Please follow the repo of [XNM](https://github.com/shijx12/XNM-Net/tree/master/exp_vqa) for:
1. Downloading and unzip [Glove Vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip)
2. Process the Glove vectors:
```
python process_glove.py
```
3. Preprocess bottom-up features

### Question generation
1. Generate question files for training
```
python preprocess/preprocess_questions.py --glove_pt ./data/glove_embedding.pickle --input_questions_json novel_vqa_train_question.json --input_annotations_json novel_vqa_train_annotation.json --output_pt ./data/train_questions_vqa.pt --vocab_json ./data/vocab_vqa.json --mode train --val_questions_json novel_vqa_val_question.json
```
2. Generate question files for validation with novel objects
```
python preprocess/preprocess_questions.py --glove_pt ./data/glove_embedding.pickle --input_questions_json novel_vqa_val_question.json --input_annotations_json novel_vqa_val_annotation.json --output_pt ./data/val_questions_vqa.pt --vocab_json ./data/vocab_vqa.json --mode val --val_questions_json novel_vqa_val_question.json
```
3. Generate question files for validation with known objects
```
python preprocess/preprocess_questions.py --glove_pt ./data/glove_embedding.pickle --input_questions_json novel_vqa_val_known_question.json --input_annotations_json novel_vqa_val_known_annotation.json --output_pt ./data/val_questions_vqa_known.pt --vocab_json ./data/vocab_vqa.json --mode val --val_questions_json novel_vqa_val_known_question.json
```

### Training
```
python train.py --input_dir ./data --feature_dir $FEAT_DIR --save_dir $SAVE_DIR --val --train_question_pt train_questions_vqa.pt --val_question_pt val_questions_vqa.pt --vocab_json vocab_vqa.json
```

### Validation
For evaluation on the novel split:
```
python validate.py --input_dir ./data --feature_dir $FEAT_DIR --ckpt $SAVE_DIR/model.pt --val_question_pt val_questions_vqa.pt --vocab_json vocab_vqa.json
```
For evaluation on the known split:
```
python validate.py --input_dir ./data --feature_dir $FEAT_DIR --ckpt $SAVE_DIR/model.pt --val_question_pt val_questions_vqa_known.pt --vocab_json vocab_vqa.json
```

### Novel-VQA Experiments
For experimenting with the Novel-VQA dataset, simply replace keyword `vqa` in the aforementioned commands with `nvqa`. For training, add additional arguments `--T_ctrl 3` and `--stack_len 4` to be consistent with previous studies.  
