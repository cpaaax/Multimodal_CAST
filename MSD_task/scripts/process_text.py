import json
import os



def process_text(word2index, text_files, attributes, img_feature_path, text_comment_file,com_num_use=5):
    all_imgs = os.listdir(img_feature_path)
    text_comment_data = json.load(open(text_comment_file, 'r'))
    del_items = ["sarcasm", "sarcastic", "reposting", "<url>", "joke", "humour", "humor", "jokes", "irony", "ironic", "exgag"]


    attributes_data = open(attributes, 'rb')
    attr_dict = {}
    for attr in attributes_data:
        content=eval(attr)
        attr_dict[content[0]]=content[1:]

    # store all data
    data_all = {}
    for text_file in text_files:
        file = open(text_file, "rb")
        for i, line in enumerate(file):
            content = eval(line)
            image = content[0]
            sentence = content[1]


            attribute_data = attributes[i]
            if image + '.npy' in all_imgs:
                del_flag = False

                for del_item in del_items:
                    if del_item in sentence:
                        del_flag = True
                        break
                if not del_flag:
                    text_comment = text_comment_data[image + '.jpg']
                    ret_texts, ret_coms = text_comment["consensus_text"], text_comment["consensus_com"]
                    cur_attr = attr_dict[image]

                    ret_texts_new = []
                    for ret_text in ret_texts:
                        # ensure each ret_text contains at least 1 words
                        if len(ret_text.split(' ')) > 0 and len(ret_texts_new) < com_num_use:
                            ret_texts_new.append(ret_text)
                    # ensure the last ret_texts_new contains 5 texts
                    if len(ret_texts_new) < com_num_use:
                        for i in range(5 - len(ret_texts_new)):
                            ret_texts_new.append(ret_texts_new[i])

                    ret_coms_new = []
                    for ret_com in ret_coms:
                        # ensure each ret_text contains at least 1 words
                        if len(ret_com.split(' ')) > 0 and len(ret_coms_new) < com_num_use:
                            ret_coms_new.append(ret_com)
                    # ensure the last ret_texts_new contains 5 texts
                    if len(ret_coms_new) < com_num_use:
                        for i in range(5 - len(ret_coms_new)):
                            ret_coms_new.append(ret_coms_new[i])

                    #
                    cur_attr_index = []















if __name__ == "__main__":
    word2index = json.load(open('../construct_word_embedding/data/sarcasm_vocab.json', 'r'))
    #process train.txt, valid.txt, test.txt
    text_files = ['../text_data/train.txt', '../text_data/valid.txt', '../text_data/test.txt']
    ce_attributes = '../multilabel_database/img_to_five_words.txt'
