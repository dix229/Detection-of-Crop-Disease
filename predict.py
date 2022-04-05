from tensorflow.keras.models import Sequential
import keras 
import numpy as np
from keras.models import load_model
import time
# 保存预测结果//Save forecast results
def predict(modelpath,data,savepath):
    keras.backend.set_learning_phase(0)
    model = load_model(modelpath)
    print('load weight successfully')
    start = time.time()
    pre = model.predict_generator(
        data,
        max_queue_size=128,
        workers=8,
    )
    end = time.time()
    print('cost time:{}'.format(end-start))
    np.savetxt(savepath, pre, fmt='%f', delimiter='\n')
    print('save successfully')

# 验证结果//Validation results
def valid(labelspath,resultpath):
    labels = np.loadtxt(labelspath,dtype='int32',delimiter='\n')
    total = len(labels)
    result = np.loadtxt(resultpath, delimiter='\n').reshape(total,61)
    top_1_acc = 0
    top_5_acc = 0
    crop_type_acc = 0
    disease_type_acc = 0
    severity_wrong = 0
    severity_count = 0

    for i in range(total):
        top_1_pred = np.argmax(result[i])
        top_5_pred = result[i].argsort()[::-1][0:5]

        pred_cropClass = crop_class(top_1_pred)
        label_cropClass = crop_class(labels[i])

        crop_type_pred = pred_cropClass[0]
        crop_type_label = label_cropClass[0]

        disease_type_pred = pred_cropClass[1]
        disease_type_label = label_cropClass[1]

        if labels[i] in top_5_pred:  # top_5准确率//top_5 accuracy
            top_5_acc += 1

        if crop_type_pred == crop_type_label:  # 物种准确率//Species accuracy
            crop_type_acc += 1

        if disease_type_pred == disease_type_label:
            disease_type_acc += 1  # 病害种类准确率//Disease type accuracy rate
            severity_count += 1
            if top_1_pred == labels[i]:
                top_1_acc += 1  # top_1准确率//top_1 accuracy
            else:
                severity_wrong += 1  # 严重/正常错误率//Critical/Normal Error Rate

    print("Top_1 accuracy rate:", top_1_acc / total)
    print("Top_5 accuracy rate:", top_5_acc / total)
    print("Variety accuracy rate:", crop_type_acc / total)
    print("Accuracy rate of diseases and insect pests:", disease_type_acc / total)
#    print("Severity accuracy rate:", 1 - (severity_wrong / severity_count))

# 物种种类//Species
def crop_class(id_class):
    # 苹果//Apple
    if 0 <= id_class < 6:
        if id_class == 0:
            return 0, 0
        elif 1 <= id_class <= 2:
            return 0, 1
        elif id_class == 3:
            return 0, 2
        else:
            return 0, 3

    # 樱桃//Cherry
    if 6 <= id_class < 9:
        if id_class == 6:
            return 1, 4
        else:
            return 1, 5

    # 玉米//corn
    if 9 <= id_class < 17:
        if id_class == 9:
            return 2, 6
        elif 10 <= id_class <= 11:
            return 2, 7
        elif 12 <= id_class <= 13:
            return 2, 8
        elif 14 <= id_class <= 15:
            return 2, 9
        else:
            return 2, 10

    # 葡萄//Grape
    if 17 <= id_class < 24:
        if id_class == 17:
            return 3, 11
        elif 18 <= id_class <= 19:
            return 3, 12
        elif 20 <= id_class <= 21:
            return 3, 13
        else:
            return 3, 14

    # 柑桔//Citrus
    if 24 <= id_class < 27:
        if id_class == 24:
            return 4, 15
        else:
            return 4, 16

    # 桃//Peach
    if 27 <= id_class < 30:
        if id_class == 27:
            return 5, 17
        else:
            return 5, 18

    # 辣椒//chili
    if 30 <= id_class < 33:
        if id_class == 30:
            return 6, 19
        else:
            return 6, 20

    # 马铃薯//potato
    if 33 <= id_class < 37:
        if id_class == 33:
            return 7, 21
        elif 34 <= id_class <= 35:
            return 7, 22
        else:
            return 7, 23

    # 草莓//Strawberry
    if 37 <= id_class < 41:
        if id_class == 37:
            return 8, 24
        else:
            return 8, 25

    if 41 <= id_class < 61:
        if id_class == 41:
            return 9, 26
        elif 42 <= id_class <= 43:
            return 9, 27
        elif 44 <= id_class <= 45:
            return 9, 28
        elif 46 <= id_class <= 47:
            return 9, 29
        elif 48 <= id_class <= 49:
            return 9, 30
        elif 50 <= id_class <= 51:
            return 9, 31
        elif 52 <= id_class <= 53:
            return 9, 32
        elif 54 <= id_class <= 55:
            return 9, 33
        elif 56 <= id_class <= 57:
            return 9, 34
        elif 58 <= id_class <= 59:
            return 9, 35
        else:
            return 9, 36
