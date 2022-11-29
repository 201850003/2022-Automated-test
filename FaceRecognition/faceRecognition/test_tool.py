from read_data import read_img_list, read_name_list
from train_model import Model
import re


# 对原数据集子文件夹名称做简单处理
def handle_name_list(path):
    name_list = read_name_list(path)
    for i in range(0, len(name_list)):
        name_list[i] = name_list[i].split("pins_")[1].replace("_", " ")
    return name_list


# 打印分割线
def print_line(delim_char, times, on_top):
    if on_top:
        print(delim_char*times + "\n"*2)
    else:
        print("\n"*2 + delim_char*times)


# 获取测试数据集中所有图片的名称
def get_actual_res(name_list, path):
    res = []

    actual_name_list = read_name_list(path)
    for i in range(0, len(actual_name_list)):
        actual_name = re.split(r'\d', actual_name_list[i])[0]
        if actual_name in name_list:
            res.append(actual_name)
        else:
            res.append('not find')

    return res


# 传入待测图片所在文件夹的路径，进行测试
def test_model(name_list, img_path, target_model):
    model = Model()
    model.load(target_model)

    res = []
    # 获取模型人脸识别结果
    img_list = read_img_list(img_path)
    for img in img_list:
        index, probability = model.predict(img)
        res.append(name_list[index] if (index != -1) else 'not find')

    return res


def eval_accuracy(test_list, actual_list):
    n_r = 0                     # 正确识别人脸的数量
    n_c = len(actual_list)      # 文件夹下图片总数

    for test_name, actual_name in test_list, actual_list:
        if test_name == actual_name:
            n_r += 1

    rate = '{:.2%}'.format(n_r / n_c)
    print_line("=", 20, True)
    print("The accuracy rate is: " + rate + "\n")
    print_line("=", 20, False)


if __name__ == '__main__':
    origin_img_path = 'D:\\Local Project\\faceRec\\originDataSet'
    handle_img_path = 'D:\\Local Project\\faceRec\\2022-Automated-test\\FaceRecognition\\pictures\\dataset'
    model_path = 'D:\\Local Project\\faceRec\\2022-Automated-test\\FaceRecognition\\model.h5'

    existing_name_list = handle_name_list(origin_img_path)
    actual_res = get_actual_res(existing_name_list, handle_img_path)
    test_res = test_model(existing_name_list, handle_img_path, model_path)


