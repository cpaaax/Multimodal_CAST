import os
import json



def analyze_text_len(train_path, valid_path, test_path):
    length = {}
    for path in [train_path, valid_path, test_path]:
        with open(path, 'r') as f:
            items = f.readlines()
            for item in items:
                text = item.strip().split(',')[1].replace('\'', '')

                text_len = len(text.split(' '))
                if text_len not in length:
                    length[text_len] = 1
                else:
                    length[text_len] += 1
    lengths = sorted(length.items(), key=lambda item: item[0], reverse=False)
    lengths = dict(lengths)
    keys = list(lengths.keys())
    values = list(lengths.values())
    cnt = 0
    k = 28
    for i in range(k):
        cnt+= values[i]

    all_num = sum(values)
    print('top {} {} len  is {}'.format(k,keys[k], float(cnt/all_num)))
    print()





if __name__=="__main__":
    train_path = '../text_data/train.txt'
    valid_path = '../text_data/valid.txt'
    test_path = '../text_data/test.txt'
    analyze_text_len(train_path, valid_path, test_path)
