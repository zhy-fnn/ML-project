import numpy as np
import csv

class Data:
    def __init__(self,l,acc):
        self.l = l
        self.acc = acc

    def to_row(self):
        return [f"{self.l:.3f}",f"{self.acc:.3f}"]

class Sigma:
    def __init__(self, sca, Zsigma, Asigma, flat):
        self.sca = sca
        self.Zsigma = Zsigma
        self.Asigma = Asigma
        self.flat = flat

    def to_row(self):
        return [f"{self.sca}",f"{self.Zsigma:.3f}",f"{self.Asigma:.3f}",f"{self.flat:.3f}"]

sigma = []

"""数据集构建"""
rng = np.random.default_rng(0)
n, noise = 400, 0.1
n0, n1 = n // 2, n - n // 2

t = np.linspace(0, np.pi, n0)
X0 = np.c_[np.cos(t), np.sin(t)]
X1 = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
# print(X0.shape, X1.shape) (200, 2) (200, 2)

X = np.vstack([X0, X1]) + rng.normal(scale = noise, size = (n, 2))
# print(X.shape) (400, 2)
Y = np.r_[np.zeros(n0), np.ones(n1)].reshape(n, 1)
# print(Y.shape) (400, 1)

"""模型构建"""
h = 16
sca = [0.01,0.1,0.33,1,5]
r = np.random.default_rng(2)
for s in sca:
    w1 = r.normal(scale = s, size = (2, h))
    b1 = np.zeros(h)
    w2 = r.normal(scale = s, size = (h, 1))
    b2 = np.zeros(1)
    #print(w1.shape, b1.shape, w2.shape, b2.shape) (2, 16) (16,) (16, 1) (1,)

    """前向传播"""
    def forward(X, w1, b1, w2, b2):
        Z1 = X @ w1 + b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ w2 + b2
        A2 = 1 / (1 + np.exp(-Z2))
        # print(Z1.shape, A1.shape, Z2.shape, A2.shape)
        return A2, Z2, A1, Z1

    """损失函数"""
    def loss(A2, Y):
        l = -(1 / n) * (Y * np.log(A2 + 1e-12) + (1 - Y) * np.log(1 - A2 + 1e-12)).sum()
        return l

    """反向传播"""
    def backward(A2, A1, X, Y):
        dZ2 = (1 / n) * (A2 - Y)
        dw2 = A1.T @ dZ2
        db2 = dZ2.sum(axis = 0)
        # print(dZ2.shape, dw2.shape, db2.shape) (400, 1) (16, 1) (1,)
        dA1 = dZ2 @ w2.T
        dZ1 = dA1 * (1 - A1 ** 2)
        dw1 = X.T @ dZ1
        db1 = dZ1.sum(axis = 0)
        # print(dA1.shape, dZ1.shape, dw1.shape, db1.shape) (400, 16) (400, 16) (2, 16) (16,)
        return dw2, db2, dw1, db1

    def num_grad(parm, ix, exp = 1e-6):
        old = parm[ix]
        parm[ix] = old + exp; ll = loss(forward(X, w1, b1, w2, b2)[0], Y)
        parm[ix] = old - exp;lr = loss(forward(X, w1, b1, w2, b2)[0], Y)
        parm[ix] = old
        return (ll - lr) / (2 * exp)

    """训练过程"""
    iterator = 10000
    data = []
    lr = 0.05
    A2, Z2, A1, Z1 = forward(X, w1, b1, w2, b2)
    sigma.append(Sigma(s, Z1.std(), A1.std(), (np.abs(A1) > 0.99).mean()))
    for it in range(1, iterator + 1):
        A2, Z2, A1, Z1 = forward(X, w1, b1, w2, b2)
        l = loss(A2, Y)
        dw2, db2, dw1, db1 = backward(A2, A1, X, Y)
        w2 -= dw2 * lr; b2 -= db2 * lr; w1 -= dw1 * lr; b1 -= db1 * lr
        pred = (A2 > 0.5)
        acc = np.mean(pred == Y)
        data.append(Data(l, acc))
        if it % 1000 == 0:
            print(l, acc)


with open("sigma.csv", "w", encoding = "UTF-8-sig", newline = "") as f:
    w = csv.writer(f)
    w.writerow(["scale","Z1sigma", "A1sigma", "flat"])
    w.writerows(line.to_row() for line in sigma)

# """差分验证"""
# A2, _, A1, _ = forward(X, w1, b1, w2, b2)
# dw2, db2, dw1, db1 = backward(A2, A1, X, Y)
# print("dw2:",dw2[5, 0],num_grad(w2, (5, 0)))
# print("db2:",db2[0],num_grad(b2, 0))
# print("dw1:",dw1[1, 5],num_grad(w1, (1, 5)))
# print("db1:",db1[13],num_grad(b1, 13))
#
# """记录数据"""
# with open("MLP scale = 5.csv", "w", encoding = "UTF-8-sig", newline = "") as f:
#     w = csv.writer(f)
#     w.writerow(["loss","accuracy"])
#     w.writerows(line.to_row() for line in data)