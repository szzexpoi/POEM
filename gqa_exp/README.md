# Zero-shot VQA experiments on the GQA dataset

### Question generation
1. Download the [Glove Vector](http://nlp.stanford.edu/data/glove.840B.300d.zip)
2. Generate question files for training
```
python preprocess/preprocess_questions.py --glove_pt $GLOVE_FILE --input_questions_json novel_gqa_train_question.json --input_annotations_json novel_gqa_train_annotation.json --output_pt ./data/train_questions_gqa.pt --vocab_json ./data/vocab_gqa.json --mode train --val_questions_json novel_gqa_val_question.json
```
3. Generate question files for validation with novel objects
```
python preprocess/preprocess_questions.py --glove_pt $GLOVE_FILE --input_questions_json novel_gqa_val_question.json --input_annotations_json novel_gqa_val_annotation.json --output_pt ./data/val_questions_gqa.pt --vocab_json ./data/vocab_gqa.json --mode val --val_questions_json novel_gqa_val_question.json
```
4. Generate question files for validation with known objects
```
python preprocess/preprocess_questions.py --glove_pt $GLOVE_FILE --input_questions_json novel_gqa_val_known_question.json --input_annotations_json novel_gqa_val_known_annotation.json --output_pt ./data/val_questions_gqa_known.pt --vocab_json ./data/vocab_gqa.json --mode val --val_questions_json novel_gqa_val_known_question.json
```

### Training
```
python train.py --input_dir ./data --feature_dir $FEAT_DIR --save_dir $SAVE_DIR --val --train_question_pt train_questions_gqa.pt --val_question_pt val_questions_gqa.pt --vocab_json vocab_gqa.json
```

### Validation
For evaluation on the novel split:
```
python validate.py --input_dir ./data --feature_dir $FEAT_DIR --ckpt $SAVE_DIR/model.pt --val_question_pt val_questions_gqa.pt --vocab_json vocab_gqa.json
```
For evaluation on the known split:
```
python validate.py --input_dir ./data --feature_dir $FEAT_DIR --ckpt $SAVE_DIR/model.pt --val_question_pt val_questions_gqa_known.pt --vocab_json vocab_gqa.json
```
