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
        # 正则匹配图片名称的第一个数字，切割后取第一部分
        one_actual_name = re.split(r'\d', actual_name_list[i])[0]
        if one_actual_name in name_list:
            res.append(one_actual_name)
        else:
            res.append('not find')

    return res


# 传入待测图片所在文件夹的路径，进行测试
def test_model(name_list, img_path, predict_threshold):
    model = Model()
    model.load()

    res_name = []
    res_prob = []
    # 获取模型人脸识别结果
    img_list = read_img_list(img_path)
    for img in img_list:
        index, prob = model.predict(img)
        # 筛选预测概率过低的结果
        if index != -1 and prob >= float(predict_threshold):
            res_name.append(name_list[index])
            res_prob.append(prob)
        else:
            res_name.append("not find")
            res_prob.append(1)

    return res_name, res_prob


# 计算准确率
def eval_accuracy(test_list, actual_list, actual_name_list, prob_list, predict_threshold1, predict_threshold2):
    n_r = 0                     # 正确识别人脸的数量
    n_c = len(actual_list)      # 文件夹下图片总数
    not_find_list = []

    # print(test_list)
    for k in range(0, n_c):
        # 此处不是筛选测试结果，只是在计算准确率时用概率阈值判断模型的预测能力
        if test_list[k] == actual_list[k] and prob_list[k] >= predict_threshold2:
            if test_list[k] == actual_list[k]:
                n_r += 1
            else:
                not_find_list.append(actual_name_list[k])

    rate = n_r/n_c
    output(rate, predict_threshold1, predict_threshold2, not_find_list)


# 规范输出格式
def output(accuracy, predict_threshold1, predict_threshold2, not_find_list):
    # 这四行代码用于从0开始多次计算时的格式化输出
    # if predict_threshold == 0:
    #     print_line("=", 50, True)
    # else:
    #     print_line("*", 50, True)
    print_line("=", 50, True)
    print("The accuracy rate is: {:.2%}.".format(accuracy))
    print("The prediction threshold for test result filter is: {:.2f}".format(predict_threshold2))
    print("The prediction threshold for the accuracy calculation filter is: {:.2f}".format(predict_threshold1))
    print("The following is a list of images that identify errors: ")
    for not_find_name in not_find_list:
        print("     " + not_find_name)
    print()


if __name__ == '__main__':
    origin_img_path = '..\\pictures\\3_classes_pins_dataset'
    handle_img_path = '..\\pictures\\newdataset'

    # 获取正确答案
    existing_name = handle_name_list(origin_img_path)           # 获取人脸识别库中已录入的人脸对应人名
    actual_name = read_name_list(handle_img_path)               # 获取测试集中所有图片名称
    actual_res = get_actual_res(existing_name, actual_name)     # 获取人脸识别的标准答案

    # 设置概率阈值，获取测试结果
    predict_threshold_1 = float(input("Please enter the probability threshold for test result filter: "))
    test_res, test_res_prob = test_model(existing_name, handle_img_path, predict_threshold_1)

    # 设置概率阈值，计算准确率
    # for j in np.arange(0, 1, 0.1):
    #     eval_accuracy(test_res, actual_res, actual_name, test_res_prob, j)
    predict_threshold_2 = float(input("Please enter the probability threshold for the accuracy calculation filter: "))
    eval_accuracy(test_res, actual_res, actual_name, test_res_prob, predict_threshold_1, predict_threshold_2)

    print_line("=", 50, False)
