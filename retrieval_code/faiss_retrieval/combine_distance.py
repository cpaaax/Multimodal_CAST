import h5py
import json
import os
import numpy
from imageio import imread
import numpy as np
def combine_img_orders(order_paths, order_index_save_path):
    order_all = []
    last_all = {}
    for path in order_paths:
        data = list(json.load(open(path, 'r')).values())
        order_all.extend(data)
    for i, j in enumerate(order_all):
        last_all[str(i)] = j
    json.dump(last_all, open(order_index_save_path, 'w'))

def get_distance_mean(file_path):
    with h5py.File(file_path, 'r') as f:
        top_k_distance = f['top_100k_distance'][()]
        top_k_index = f['top_100k_index'][()]
    overall_mean = numpy.mean(top_k_distance[:, :10])
    return overall_mean, top_k_distance[:, :100000], top_k_index[:, :100000]

def get_overlap(index_img, index_text):
    size = index_img.shape[0]
    overlap_results = []
    for i in range(size):
        cur_index_img = index_img[i]
        cur_index_text = index_text[i]
        # print(set(cur_index_img)&set(cur_index_text))
        overlap_results.append(set(cur_index_img)&set(cur_index_text))
    return overlap_results

def get_overlap_for_broken(index_img, index_text, img_status):
    size = index_img.shape[0]
    overlap_results = []
    for i in range(size):
        img_ = img_status[i]
        cur_index_img = index_img[i]
        cur_index_text = index_text[i]
        if not img_: # when img is good, load the overlap results, if img is bad, just load the text results

            overlap_results.append(set(cur_index_img) & set(cur_index_text))
        else:
            overlap_results.append(set(cur_index_text))
        # print(set(cur_index_img)&set(cur_index_text))
    return overlap_results

def weighted_similar_results(index_img, index_text, dist_img, dist_text, overlap_results, img_dis_weight, text_dis_weight):
    weighted_dist_img = dist_img*img_dis_weight
    weighted_dist_text = dist_text * text_dis_weight
    last_result = []
    for i, overlap_result in enumerate(overlap_results):
        temp_result = []
        cur_index_img = index_img[i]
        cur_index_text = index_text[i]

        # change the data type from set to list
        overlap_result = list(overlap_result)
        for idx in overlap_result:
            # get the index in cur_index_img which is equal to the idx in overlap_result
            # tmp_idx_img = cur_index_img.index(idx)
            tmp_idx_img = numpy.where(cur_index_img == idx)
            tmp_idx_text = numpy.where(cur_index_text == idx)

            # obtain the distance based on the index
            tmp_dis_img = weighted_dist_img[i, tmp_idx_img].squeeze()
            tmp_dis_text = weighted_dist_text[i, tmp_idx_text].squeeze()
            temp_result.append(tmp_dis_img+tmp_dis_text)
        temp_result = numpy.array(temp_result)
        sorted_index = temp_result.argsort()
        sorted_result = [overlap_result[i] for i in sorted_index]
        last_result.append(sorted_result[:20])
    return last_result

def weighted_similar_results_for_broken(index_img, index_text, dist_img, dist_text, overlap_results,
                                        img_dis_weight, text_dis_weight, img_status):
    weighted_dist_img = dist_img * img_dis_weight
    weighted_dist_text = dist_text * text_dis_weight
    last_result = []
    for i, overlap_result in enumerate(overlap_results):
        temp_result = []
        cur_index_img = index_img[i]
        cur_index_text = index_text[i]
        img_s = img_status[i]
        # change the data type from set to list
        overlap_result = list(overlap_result)
        for idx in overlap_result:
            # get the index in cur_index_img which is equal to the idx in overlap_result
            # tmp_idx_img = cur_index_img.index(idx)
            tmp_idx_img = numpy.where(cur_index_img == idx)
            tmp_idx_text = numpy.where(cur_index_text == idx)

            # obtain the distance based on the index
            tmp_dis_img = weighted_dist_img[i, tmp_idx_img].squeeze()
            tmp_dis_text = weighted_dist_text[i, tmp_idx_text].squeeze()
            if not img_s: # if img quality is good, using the combined distance
                temp_result.append(tmp_dis_img+tmp_dis_text)
            else:  # if img quality is bad, just using the text distance
                temp_result.append(tmp_dis_text)
        temp_result = numpy.array(temp_result)
        sorted_index = temp_result.argsort()
        sorted_result = [overlap_result[i] for i in sorted_index]
        last_result.append(sorted_result[:10])
    return last_result










def convert_idx2img(ret_results, order_idx_path, ret_result_save_path, query_order):
    order_idx = json.load(open(order_idx_path, 'r'))
    ret_imgs = {}
    for i, result in enumerate(ret_results):
        query_id = query_order[i]
        tmp_imgs = []
        for j in result:
            tmp_imgs.append(order_idx[str(j)])  # format: {img_id:date}
        ret_imgs[query_id] = tmp_imgs
    json.dump(ret_imgs, open(ret_result_save_path, 'w'))

if __name__=='__main__':
    # first get the index of all data
    use_years = ['2014','2015', '2016', '2017', '2017_new', '2018', '2018_new', '2019', '2019_new']
    order_root = '../image_fc_feature_combine'
    order_index_save_path = os.path.join(order_root, 'order_all.json')
    order_paths = [os.path.join(order_root, year, year+'_order_all.json') for year in use_years]
    print('generate order_all.json')
    combine_img_orders(order_paths, order_index_save_path)

    # combine the distance
    text_similar_file = './faiss_save/ITR/similar_results_on_all_texts.h5'
    img_similar_file = './faiss_save/ITR/similar_results_on_all_imgs.h5'
    result_save_path = './faiss_save/ITR/retrieval_results_index.json'

    query_order_path = './ITR/ITR_text.json'
    # img_order = './NER/NER_2015/ner_2015_img_order.json'
    query_data = list(json.load(open(query_order_path, 'r')).values())
    query_order = [i['id']+'.jpg' for i in query_data]

    broken_img = True
    img_dis_mean, img_dist, img_index = get_distance_mean(img_similar_file)
    text_dis_mean, text_dist, text_index = get_distance_mean(text_similar_file)
    img_dis_weight = text_dis_mean / (img_dis_mean + text_dis_mean)
    text_dis_weight = img_dis_mean / (img_dis_mean + text_dis_mean)
    print('img_dis_mean: ', img_dis_mean, 'text_dis_mean: ', text_dis_mean, 'img_dis_weight: ', img_dis_weight,
          'text_dis_weight: ', text_dis_weight, )
    overlap_results = get_overlap(img_index, text_index)
    last_retrieval_result = weighted_similar_results(img_index, text_index, img_dist, text_dist, overlap_results, img_dis_weight, text_dis_weight)

    assert len(query_order) == len(last_retrieval_result)
    convert_idx2img(last_retrieval_result, order_index_save_path, result_save_path, query_order)






