import numpy
import faiss
import h5py
import os
import math
import torch
from torchvision import transforms as trn
from imageio import imread
import gc
import sys
sys.path.append("..")
sys.path.append(".")
import json
from extract_img_features.misc.resnet_utils import myResnet
from extract_img_features.misc import resnet  as resnet
import numpy as np
import time
from simcse import SimCSE


preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_ptr(data_size):
    data_ptr = [0]
    for i, data_num in enumerate(data_size):
            data_ptr.append(data_ptr[-1]+data_num)
    print(data_ptr)
    return data_ptr

def write_index(data, years, data_size, n_cluster, m, bits, index_train_path, index_add_path, train=False):
    data_all_num = sum(data_size)
    data_all_ptr = get_ptr(data_size)
    img_all = np.empty((data_all_num, 768), dtype='float32')
    for i, year in enumerate(years):
        feature_path = os.path.join(data, year, year+ '_feature_all.h5')
        with h5py.File(feature_path, 'r') as f:
            img_tmp = f[year][()]
        print('There are {} texts in year {}, the dimension is {}'.format(img_tmp.shape[0],
                                                                         year, img_tmp.shape[1]))
        img_all[data_all_ptr[i]:data_all_ptr[i+1]] = img_tmp

    img_num = img_all.shape[0]
    d = img_all.shape[1]
    print('There are {} texts in total, the dimension is {}'.format(img_num, d))
    # faiss.normalize_L2(imgs)

    if train:
        quantizer = faiss.IndexFlatL2(d)  # def the method of calculating distance (L2 distance, here)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, n_cluster, m, bits)  # construct the index
        gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
        )
        print('start training faiss')
        train_s = time.time()
        gpu_index.train(img_all)                       # train the index on the data
        train_e = time.time()
        print('finish trianing faiss, time: {}'.format(train_e - train_s))
        print('start writing training index')
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        faiss.write_index(cpu_index, index_train_path)  # save the img index
    else:
        if os.path.exists(index_train_path):
            cpu_index = faiss.read_index(index_train_path)
            gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
                cpu_index
            )
            print('starting adding data')
            add_s = time.time()
            gpu_index.add(img_all)
            add_e = time.time()
            print('finish adding faiss, time: {}'.format(add_e - add_s))
            print('start writing faiss')
            cpu_index = faiss.index_gpu_to_cpu(gpu_index)
            faiss.write_index(cpu_index, index_add_path)  # save the img index



def get_distance_orig_index(sorted_distance, sorted_index):
    # get original index
    indices_ori = np.argsort(sorted_index, axis=1)
    # get original distance
    d_ori = np.take_along_axis(sorted_distance, indices_ori,axis=1)
    return d_ori



def search(index_path, encoder_model, query_texts):
    # extract text feature of query text
    with torch.no_grad():
        text_feature = encoder_model.encode(query_texts, batch_size=768)

    query_feats = text_feature.data.cpu().float().numpy()

    # faiss.normalize_L2(query_feats)
    # load the trained index and search similar index
    print('load index')
    index = faiss.read_index(index_path)

    index.nprobe = 10000   # nprobe is the number of nearby cells to search

    print('start searching')
    start = time.time()
    # D, I = index.search(query_feats, k=10)
    Distance, Index = index.search(query_feats, k=100000)

    end = time.time()
    print('time {}'.format(end-start))
    print('distance of nearest imgs:, ', Distance[:5])
    print('index of nearest imgs:, ', Index[:5])

    with h5py.File('./rel/similar_results_on_all_texts.h5', 'w') as f:
        # to reduce the storage space, we use float16 rather than float32
        # additionally, we only save top-100000 similar results
        f.create_dataset('top_100k_distance', data=np.float16(Distance/100))
        f.create_dataset('top_100k_index', data=Index)



def obtain_data_size(data_root, use_years):
    data_size = []
    for year in use_years:
        year_order_path = json.load(open(os.path.join(data_root, year, '{}_order_all.json'.format(year)), 'r'))
        cur_data_size = len(year_order_path)
        data_size.append(cur_data_size)
    return data_size


if __name__=="__main__":
    data_root = '../text_fc_feature_combine'
    use_years = ['2014', '2015', '2016', '2017', '2017_new', '2018', '2018_new', '2019', '2019_new']
    data_size = obtain_data_size(data_root, use_years)

    index_root_path = './faiss_save'
    if not os.path.exists(index_root_path):
        os.makedirs(index_root_path)

    index_train_path = os.path.join(index_root_path, 'total_text_train_full.index')
    index_add_path = os.path.join(index_root_path, 'total_text_add_full.index')


    faiss.omp_set_num_threads(80)

    n_cluster = 270000
    m = 8  # number of centroid IDs in final compressed vectors. (In other words, each sentence_embedding vector is split into m parts.) The value of m must can be divided evenly by d.
    bits = 8  # number of bits in each centroid

    # for training data
    write_index(data_root, use_years, data_size, n_cluster, m, bits, index_train_path, index_add_path, train=True)
    # for adding data
    write_index(data_root, use_years,data_size, n_cluster, m, bits, index_train_path, index_add_path, train=False)


    query_order_path = './ITR/ITR_text.json'

    query_data = list(json.load(open(query_order_path, 'r')).values())
    text_data = [i['text'] for i in query_data]

    # def the encoder for extracting features of query text
    encoder_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    # search similar texts
    search(index_add_path, encoder_model, text_data)
