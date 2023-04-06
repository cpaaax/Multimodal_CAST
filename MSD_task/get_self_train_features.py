import os
import shutil
from tqdm import tqdm
if __name__ == "__main__":
    target_num = 3

    source_path = '/home/pc/Desktop/EMNLP_data/sarcasm/extract_features_resnet152_10'
    target_path = '/home/sdb_pro/EMNLP2022_self-training_code/self_training_data/sarcasm/extract_features_resnet152_3'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    features = os.listdir(source_path)
    for feature in tqdm(features):
        if '-0' in feature or '-1' in feature or '-2' in feature:
            shutil.copyfile(os.path.join(source_path, feature), os.path.join(target_path, feature))

