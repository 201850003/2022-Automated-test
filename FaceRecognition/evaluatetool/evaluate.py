import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def smooth_xy(figures,points):#制作散点图
    x=np.array(figures)
    y=np.array(points)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth,y_smooth]


if __name__ == '__main__':
    #以下数据由训练模型时该标签数的最佳训练次数和样本下得到的预测准确率
    xy_s=smooth_xy(figures=[2,3,4,8,10,15,20,30,40,50,60,70,80,90,105],points=[0.7608,0.6000,0.4444,0.3855,0.3400,0.2985,0.2755,0.2500,0.2405,0.2335,0.2285,0.2231,0.2190,0.2155,0.2121])
    plt.plot(xy_s[0],xy_s[1])
    plt.savefig("../doc/evaluate.png",format="png",dpi=1000)