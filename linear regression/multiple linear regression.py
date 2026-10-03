import numpy as np

data=[]

numbers=1000
"""数据集获取"""
np.random.seed(0)
x=np.random.uniform(0,100,(numbers,100)) #有一千个方程，每个方程的参数有100个
bios=np.random.randn(numbers,)*0.5
theta_r=np.random.randint(0,5,(99,)) #假设有100个参数
theta_r=np.r_[np.ones(1,),theta_r] #让其中一个参数充当b

"""标准化"""
x_mean,x_std=x.mean(),x.std()
x_s=(x-x_mean)/x_std

"""训练集"""
y=x_s@theta_r+bios

"""预测"""
def predict(theta,x_s):
    return x_s@theta

"""损失函数"""
def loss(theta,x_s,y):
    return (1/numbers)*((predict(theta,x_s)-y)**2).sum()

"""梯度下降"""
def grad(theta,x_s,y):
    y_hat=predict(theta,x_s)
    dtheta=(2/numbers)*x_s.T@(y_hat-y)
    return dtheta

"""训练过程"""
losses=[]
lr=0.05
theta=np.zeros(100,)
for i in range(1000):
    l=loss(theta,x_s,y)
    dtheta=grad(theta,x_s,y)
    theta-=dtheta*lr
    losses.append(l)
    if np.linalg.norm(dtheta)<1e-10:
        data.append(i)
        break

"""对拍"""
print(np.allclose(theta_r,theta,atol=1e-1))
print(losses[-1])