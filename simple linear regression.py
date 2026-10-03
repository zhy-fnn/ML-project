import numpy as np
import matplotlib.pyplot as plt
import csv
import time

class Data:
    def __init__(self,times,number,w_r,b_r,w,b,loss_end,val):
        self.times=times
        self.number=number
        self.w_r=w_r
        self.b_r=b_r
        self.w=w
        self.b=b
        self.loss_end=loss_end
        self.val=val

    def to_row(self):
        return [self.times, self.number,
                f"{self.w_r:.3f}", f"{self.b_r:.3f}",
                f"{self.w:.3f}", f"{self.b:.3f}",
                f"{self.loss_end:.3f}", f"{self.val:.3e}"]

data=[]
for numbers in (10,20,50,100,1000,10000,50000,100000):
    """数据集构建"""
    np.random.seed(0)
    x=np.random.uniform(0,100,numbers)
    w_r=np.random.randint(0,5)
    b_r=np.random.randint(0,5)
    # bios=np.random.uniform(0,1e-2,numbers)
    bios=np.random.randn(numbers)*0.5

    """标准化"""
    x_mean,x_std=x.mean(),x.std()
    x_s=(x-x_mean)/x_std

    """训练集"""
    y=x*w_r+b_r+bios

    """预测"""
    def predict(w,x_s,b):
        return w*x_s+b

    """损失函数(MSE)"""
    def loss(w,x_s,b,y):
        return (1/numbers)*((predict(w,x_s,b)-y)**2).sum()

    """损失函数(MAE)"""
    def lossc(w,x_s,b,y):
        return (1/numbers)*abs(predict(w,x_s,b)-y).sum()

    """梯度下降（公式版）"""
    def grads(w,x_s,b,y):
        y_hat=predict(w,x_s,b)
        dw=(2/numbers)*((y_hat-y)*x_s).sum()
        db=(2/numbers)*(y_hat-y).sum()
        return dw,db

    """梯度下降（差分版）"""
    def gradsc(w,x_s,b,y):
        h=np.random.uniform(0,1e-3)
        dw=(loss(w+h,x_s,b,y)-loss(w-h,x_s,b))/(2*h)
        return dw

    """训练过程"""
    losses=[]
    lr=0.05
    w=0
    b=0
    times=None
    for i in range(1000):
        l=loss(w,x_s,b,y)
        dw,db=grads(w,x_s,b,y)
        w-=dw*lr
        b-=db*lr
        losses.append(l)
        if np.linalg.norm([dw,db])<1e-12:
            times=i
            break

    """验证泛化能力"""
    xte=np.random.uniform(0,100,100)
    bios=np.random.randn(100)*0.5
    xte_s=(xte-x_mean)/x_std
    yte=w_r*xte_s+b_r+bios
    val=np.mean(loss(w,xte_s,b,yte))
    data.append(Data(times,numbers,w_r,b_r,w,b,losses[-1],val))


"""写入csv"""
with open("simple linear regression.csv",'w',encoding="UTF-8-sig",newline='') as f:
    w=csv.writer(f)
    w.writerow(["times","numbers","w_real","b_real","w","b","loss_end","value"])
    w.writerows(line.to_row() for line in data)

"""还差检测不同数据量下模型的泛化能力，用差分法求导数的情况（h从大到小每次除十），以及非MSE时的训练情况"""