**Extract features**\

***Extract image features***
Download the resnet weights from [here](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21038672r_connect_polyu_hk/EfDWDBJUTLlBkBqpJJje8VkB5of73Jc2k7RYbmiCRGLKpw?e=Znf7P0), and put the weights in `./extract_img_features/imagenet_weights/resnet101_caffe.pth`. Then run:
```
cd extract_img_features
python multi_image_preprocess.py
```
***Extract text features***
```
cd extract_text_features
python get_text_feature.py
```
**Combine the extracted features based on the year**\
For images,
```
cd combine_features
python load_img_data.py
```
A `year_feature_all.h5` and `year_order_all.json` would be generated. The `year_feature_all.h5` file stores the features, while `year_order_all.json` stores the orders of
features in `year_feature_all.h5`.

For texts,
```
cd combine_features
python load_text_data.py
```
A `year_feature_all.h5` and `year_order_all.json` are generated. The feature order in `year_feature_all.h5` is the same as the image.

***Faiss Retrieval***\
Train the faiss, and obtain the retrieval results. Here we take the ITR task as example (for other tasks, after you train the faiss, you could directly use the trained faiss index to obtain the retrieval results):

For images,
```
cd faiss_retrieval
python faiss_image_retrieval.py
```
The trained faiss index for image is saved in `./faiss_save/total_img_train_full.index`. The retrieval results of ITR is saved in `./ITR/similar_results_on_all_imgs.h5` which contains the distance and idx of the top-100000 similar images.

For texts,
```
cd faiss_retrieval
python faiss_text_retrieval.py
```
The trained faiss index for text is saved in `./faiss_save/total_text_train_full.index`. The retrieval results is saved in `./ITR/similar_results_on_all_texts.h5` which contains the distance and idx of the top-100000 similar texts. Note that the text order of retrieval results is the same as image order.


**Combine the distance**\
Combine the similar image distance and text distance, and generate the final similar multimodal tweet (i.e., image+text) results.
```
cd faiss_retrieval
python combine_distance.py
```
The final similar results are saved in `'./faiss_save/ITR/retrieval_results_index.json'`.

To get the related images, texts and comments based on the `'./faiss_save/ITR/retrieval_results_index.json'`, run:
```
cd faiss_retrieval
python get_self_training_data.py.py
```


