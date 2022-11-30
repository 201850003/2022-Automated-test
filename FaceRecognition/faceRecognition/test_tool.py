from read_data import read_img_list, read_name_list
from train_model import Model
import re
import numpy as np


# 对原数据集子文件夹名称做简单处理
def handle_name_list(path):
    name_list = read_name_list(path)
    for i in range(0, len(name_list)):
        name_list[i] = name_list[i].split("pins_")[1].replace("_", " ")
    return name_list


# 打印分割线
def print_line(delim_char, times, on_top):
    if on_top:
        print(delim_char*times + "\n")
    else:
        print(delim_char*times)


# 获取测试数据集中所有图片的名称
def get_actual_res(name_list, actual_name_list):
    res = []

    for i in range(0, len(actual_name_list)):
        one_actual_name = re.split(r'\d', actual_name_list[i])[0]
        if one_actual_name in name_list:
            res.append(one_actual_name)
        else:
            res.append('not find')

    return res


# 传入待测图片所在文件夹的路径，进行测试
def test_model(name_list, img_path):
    model = Model()
    model.load()

    res_name = []
    res_prob = []
    # 获取模型人脸识别结果
    img_list = read_img_list(img_path)
    for img in img_list:
        index, prob = model.predict(img)
        res_name.append(name_list[index] if (index != -1) else 'not find')
        res_prob.append(prob)

    return res_name, res_prob


# 计算准确率
def eval_accuracy(test_list, actual_list, actual_name_list, prob_list, predict_threshold):
    n_r = 0                     # 正确识别人脸的数量
    n_c = len(actual_list)      # 文件夹下图片总数
    not_find_list = []

    for k in range(0, n_c):
        if test_list[k] == actual_list[k] and prob_list[k] >= predict_threshold:
            n_r += 1
        else:
            not_find_list.append(actual_name_list[k])

    rate = n_r/n_c
    output(rate, predict_threshold, not_find_list)


# 规范输出格式
def output(accuracy, predict_threshold, not_find_list):
    if predict_threshold == 0:
        print_line("=", 50, True)
    else:
        print_line("*", 50, True)
    print("The accuracy rate is: {:.2%}.".format(accuracy))
    print("The prediction threshold is: {:.2f}".format(predict_threshold))
    print("The following is a list of images that identify errors: ")
    for not_find_name in not_find_list:
        print(not_find_name)


if __name__ == '__main__':
    origin_img_path = 'D:\\大三上学期课件\\FaceRecognition\\FaceRecognition\\input\\pins-face-recognition\\105_classes_pins_dataset'
    handle_img_path = '..\\pictures\\dataSet'

    existing_name = handle_name_list(origin_img_path)   # 获取人脸识别库中已录入的人脸对应人名
    actual_name = read_name_list(handle_img_path)       # 获取测试集中所有图片名称
    actual_res = get_actual_res(existing_name, actual_name)                 # 获取人脸识别的标准答案
    test_res, test_res_prob = test_model(existing_name, handle_img_path)    # 获取模型测试结果

    for j in np.arange(0.3, 0.7, 0.1):
        eval_accuracy(test_res, actual_res, actual_name, test_res_prob, j)

    print_line("=", 50, False)
