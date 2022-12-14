# Multimodal self-training  
PyTorch Code for the following paper at EMNLP2022:  
**Title**: Borrowing Human Senses: Comment-Aware Self-Training for
Social Media Multimodal Classification  
**Authors**: Chunpu Xu, Jing Li\
**Institute**: PolyU\
**Abstract**  
Social media is daily creating massive multimedia content with paired image and text, 
presenting the pressing need to automate the vision and language understanding for various 
multimodal classification tasks. Compared to the commonly researched visual-lingual data, 
social media posts tend to exhibit more implicit image-text relations.
To better glue the cross-modal semantics therein, we capture hinting features from user comments, 
which are retrieved via jointly leveraging visual and lingual similarity.
Afterwards, the classification tasks are explored via self-training in a teacher-student framework,
motivated by the usually limited labeled data scales in existing benchmarks.  
Substantial experiments are conducted on four multimodal social media benchmarks for image-text 
relation classification, sarcasm detection, sentiment classification, and hate speech detection.
The results show that our method further advances the performance of previous state-of-the-art models, 
which do not employ comment modeling or self-training.


**Framework illustration**\
![avatar](model.png)

## Data
### ITR task
Fot the ITR task, you could find the training file which contains tweet text and annotated label from `scripts/split_data/train_file.json`. 
The retrieved comments and similar tweet texts is stored in `retrieval_results_text_comment_top20.json`. 
For raw tweet image data of the ITR dataset (labeled data), please find it from [here](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21038672r_connect_polyu_hk/EbK0jcZ7bkRJrOURLgL8cj8BioN6G84fN1f1qrX0tgaQ-Q?e=9BgUhQ). 
For raw images of retrieved similar tweets (unlabeled data), please find it from [here](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21038672r_connect_polyu_hk/Eaud2hCia5pMklF-Q4TWKqUBaw9t3o_MymLlCmRcJdXGBg?e=J7XDrU)
Then, extract the image features of both labeled data and unlabeled data by the following command:
```
cd ITR_task/extract_img_features
python extract_img_feature.py
```
TODO: For the 27M dataset, the code and data for other tasks will be released soon.
## Installation
```
# Create environment
conda create -n multimodal_ST  python==3.6
# Install pytorch 
conda install -n multimodal_ST  -c pytorch pytorch==1.10.0 torchvision
```

## Training
### ITR task
```
cd ITR_task
python run_rel.py --img_feature_path /home/sdb_pro/EMNLP2022_self-training_code/training_features/img_text_rel/img_features --self_img_feature_path /home/sdb_pro/EMNLP2022_self-training_code/training_features/img_text_rel/extract_features_resnet152_3
```
We provide our pretrained models of ITR task in [here](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21038672r_connect_polyu_hk/ERzpdx2oPPtBtBRDzc3Dqk0BK3ZuBeW_QtS8BabwGvkKgg?e=nYDglq).


# License
This project is licensed under the terms of the MIT license. 