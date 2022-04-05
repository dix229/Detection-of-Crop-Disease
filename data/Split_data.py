import json
import os
import shutil
from tqdm import tqdm

try:
    for i in range(0, 61):
        os.mkdir("/Users/dixit/Study Material/Dissertation/Main/Code/Detection of Crop Disease/data/valid/" + str(i))  #put destination directory
except:
    pass

file_train = json.load(open("AgriculturalDisease_train_annotations.json", "r", encoding="utf-8"))
file_val = json.load(open("AgriculturalDisease_validation_annotations.json", "r", encoding="utf-8"))        #put both file with python files



for file in tqdm(file_train):
    filename = file["image_id"]
    origin_path = "/Users/dixit/Study Material/Dissertation/Main/Code/Detection of Crop " \
                  "Disease/data/ai_challenger_pdr2018_trainingset_20181023/AgriculturalDisease_trainingset/images/" +\
                  filename                                     #put origin path where train data exist
    ids = file["disease_class"]
    if ids == 44:
        continue
    if ids == 45:
        continue
    if ids > 45:
        ids = ids
    save_path = "/Users/dixit/Study Material/Dissertation/Main/Code/Detection of Crop Disease/data/train/" + str(ids) + "/"  #put destination directory same as os.mkdir
    shutil.copy(origin_path, save_path)

for file in tqdm(file_val):
    filename = file["image_id"]
    origin_path = "/Users/dixit/Study Material/Dissertation/Main/Code/Detection of Crop Disease/data/ai_challenger_pdr2018_validationset_20181023/AgriculturalDisease_validationset/images/" +\
                  filename                                  #put origin path where validationset data exist
    ids = file["disease_class"]
    if ids == 44:
        continue
    if ids == 45:
        continue
    if ids > 45:
        ids = ids
    save_path = "/Users/dixit/Study Material/Dissertation/Main/Code/Detection of Crop Disease/data/valid/" + str(
        ids) + "/"                                         #put destination directory same as os.mkdir
    shutil.copy(origin_path, save_path)
