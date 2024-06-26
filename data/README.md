# Data Preprocessing

### Download VQA datasets
If you wish to work on datasets used in our experiments, simply download our processed data: [link](https://drive.google.com/file/d/1dTg5Dn1BmCiwY_gXOG06lo60LOZJrw29/view?usp=sharing). Unzip the file and place the annotations in the directories for corresponding experiments, for example, `../gqa_exp/data` for the GQA experiments

Otherwise, if you want to construct your own datasets for zero-shot VQA experiments:

For VQA dataset:
1. Download the [VQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html).
2. Download instance annotations (2014) from [MSCOCO](https://cocodataset.org/#download).
3. Call the following command:
```
mkdir ../vqa_exp/data
python ./create_split_vqa.py --coco_dir $COCO_DIR --vqa_dir $VQA_DIR --save_dir $../vqa_exp/data
```
The code will reconstruct the training and validation sets based on 10 randomly selected concepts (novel concepts)

For GQA dataset:
1. Download the [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html).
2. Download simplified version of [GQA program](https://drive.google.com/file/d/1EkdqgVg562LTidWc8F9vkDkCRRCFxVGb/view?usp=sharing).
3. Call the following command:
```
python ./create_split_gqa.py --question_dir $QUESTION_DIR --scene_graph_dir $SCENE_GRAPH_DIR --semantic_dir $SEMANTIC_DIR --save_dir ./
```
4. Convert the newly generated annotations into VQA style:
```
mkdir ../gqa_exp/data
python ./gqa2vqa.py --input_dir ./ --save_dir ../gqa_exp/data
```


### Download bottom-up features
1. [for VQA](https://github.com/peteanderson80/bottom-up-attention)
2. [for GQA](https://github.com/airsplay/lxmert)

Unzip the TSV file with the following commands for VQA and GQA (note that you need to run the code with Python2), respectively:
```
python ../vqa_exp/preprocess/preprocess_features.py --input_tsv_folder $TSV_FILE_DIR --output_h5 $FEATURE_DIR/trainval_feature.h5

```
```
python2 ./extract_gqa.py --input $TSV_FILE --output $FEATURE_DIR
```
