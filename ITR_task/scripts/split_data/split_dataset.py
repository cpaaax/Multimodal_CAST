import json
import random


def split_data(data, save_file):
    data_out = {}
    train_ratio, val_ratio = 0.8, 0.1
    train_num = len(data.keys())*train_ratio



    # third img_text task
    data_img_text_task_0 = []
    data_img_text_task_1 = []
    data_img_text_task_2 = []
    data_img_text_task_3 = []


    for key, value in data.items():
        id = key
        text = value["text"]
        labels = value["label"]
        labels_split = labels.split(' ')
        img_text_task_label = labels_split[2:]

        for ptr, i in enumerate(img_text_task_label):
            if i=="1":
                label = ptr



        if label == 0:
            data_img_text_task_0.append({'id': id,'text': text, 'label':img_text_task_label})
        elif label == 1:
            data_img_text_task_1.append({'id': id,'text': text, 'label':img_text_task_label})
        elif label == 2:
            data_img_text_task_2.append({'id': id,'text': text, 'label':img_text_task_label})
        else:
            data_img_text_task_3.append({'id': id,'text': text, 'label':img_text_task_label})
    print('there are label {} 0 , {} 1, {} 2, {} 3'.format(len(data_img_text_task_0), len(data_img_text_task_1),
                                                           len(data_img_text_task_2), len(data_img_text_task_3)))

    random.shuffle(data_img_text_task_0)
    random.shuffle(data_img_text_task_1)
    random.shuffle(data_img_text_task_2)
    random.shuffle(data_img_text_task_3)
    train_img_text_0 = data_img_text_task_0[:int(train_ratio * len(data_img_text_task_0))]
    val_img_text_0 = data_img_text_task_0[int(train_ratio * len(data_img_text_task_0)):int((train_ratio+val_ratio) * len(data_img_text_task_0))]
    test_img_text_0 = data_img_text_task_0[int((train_ratio+val_ratio) * len(data_img_text_task_0)):]

    train_img_text_1 = data_img_text_task_1[:int(train_ratio * len(data_img_text_task_1))]
    val_img_text_1 = data_img_text_task_1[int(train_ratio * len(data_img_text_task_1)):int(
        (train_ratio + val_ratio) * len(data_img_text_task_1))]
    test_img_text_1 = data_img_text_task_1[int((train_ratio + val_ratio) * len(data_img_text_task_1)):]

    train_img_text_2 = data_img_text_task_2[:int(train_ratio * len(data_img_text_task_2))]
    val_img_text_2 = data_img_text_task_2[int(train_ratio * len(data_img_text_task_2)):int(
        (train_ratio + val_ratio) * len(data_img_text_task_2))]
    test_img_text_2 = data_img_text_task_2[int((train_ratio + val_ratio) * len(data_img_text_task_2)):]

    train_img_text_3 = data_img_text_task_3[:int(train_ratio * len(data_img_text_task_3))]
    val_img_text_3 = data_img_text_task_3[int(train_ratio * len(data_img_text_task_3)):int(
        (train_ratio + val_ratio) * len(data_img_text_task_3))]
    test_img_text_3 = data_img_text_task_3[int((train_ratio + val_ratio) * len(data_img_text_task_3)):]


    test_img_text_0.extend(test_img_text_1)
    test_img_text_0.extend(test_img_text_2)
    test_img_text_0.extend(test_img_text_3)

    train_img_text_0.extend(train_img_text_1)
    train_img_text_0.extend(train_img_text_2)
    train_img_text_0.extend(train_img_text_3)

    val_img_text_0.extend(val_img_text_1)
    val_img_text_0.extend(val_img_text_2)
    val_img_text_0.extend(val_img_text_3)

    random.shuffle(train_img_text_0)
    random.shuffle(val_img_text_0)
    random.shuffle(test_img_text_0)

    final_dataset = {'train': train_img_text_0, 'val': val_img_text_0,
                              'test': test_img_text_0}
    print('There are {} for training, {} for val, {} for testing'.format(len(train_img_text_0), len(val_img_text_0), len(test_img_text_0)))








    json.dump(final_dataset, open(save_file, 'w'))






if __name__ == "__main__":
    file = 'data_original.json'
    save_file = 'train_file.json'
    data = json.load(open(file, 'r'))
    random.seed(24)
    split_data(data, save_file)