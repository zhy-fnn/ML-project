import numpy as np
import matplotlib.pyplot as plt
import csv
import time

class Data:
    def __init__(self,times,number,loss_end,val,diff):
        self.times=times
        self.number=number
        self.loss_end=loss_end
        self.val=val
        self.diff=diff

    def to_row(self):
        return [self.times, self.number,
                f"{self.loss_end:.3f}", f"{self.val:.3e}",
                f"{self.diff:.3f}"]

class Atl:
    def __init__(self,i,ok):
        self.i=i
        self.ok=ok

    def to_row(self):
        return [self.i,self.ok]

class timeg:
    def __init__(self,times,func,diff):
        self.times=times
        self.func=func
        self.diff=diff

    def __str__(self):
        return str(self.to_row())

    def to_row(self):
        return [self.times,f"{self.func:.3f}",f"{self.diff:.3f}"]

"""预测"""
def predict(w,x_s,b):
    return w*x_s+b

"""损失函数(MSE)"""
def loss(w,x_s,b,y):
    return ((predict(w,x_s,b)-y)**2).mean()

"""损失函数(MAE)"""
def lossc(w,x_s,b,y):
    return abs(predict(w,x_s,b)-y).mean()

"""梯度下降（公式版）"""
def grads(w,x_s,b,y):
    y_hat=predict(w,x_s,b)
    dw=(2/numbers)*((y_hat-y)*x_s).sum()
    db=(2/numbers)*(y_hat-y).sum()
    return dw,db

"""梯度下降（差分版）"""
def gradsc(w,x_s,b,y):
    h=1e-5
    dw=(loss(w+h,x_s,b,y)-loss(w-h,x_s,b,y))/(2*h)
    db=(loss(w,x_s,b+h,y)-loss(w,x_s,b-h,y))/(2*h)
    return dw,db

"""损失函数（三次）"""
def lossthree(w,x_s,b,y):
    return (1/numbers)*((predict(w,x_s,b)-y)**3).sum()

"""梯度下降（公式三次）"""
def gradthree(w,x_s,b,y):
    y_hat=predict(w,x_s,b)
    dw=3*np.mean(x_s*(y_hat-y)**2).sum()
    db=(3/numbers)*((y_hat-y)**2).sum()
    return dw,db

"""梯度下降（差分三次）"""
def gradcthree(w,x_s,b,y,h):
    dw=(lossthree(w+h,x_s,b,y)-lossthree(w-h,x_s,b,y))/(2*h)
    db=(lossthree(w,x_s,b+h,y)-lossthree(w,x_s,b-h,y))/(2*h)
    return dw,db

order=int(input("----输入一个数----"))
while order!=-1:
    match order:
        case 1:
            numbers=1000
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
            y=x_s*w_r+b_r+bios

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
            print(f"times,numbers,w_r,b_r,w,b,losses,value")
            print([times,numbers,w_r,b_r,w,b,losses[-1],val])

            """loss曲线"""
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
            plt.rcParams["axes.unicode_minus"] = False

            X = np.c_[x_s, np.ones(numbers)]
            ols = np.linalg.lstsq(X, y, rcond=None)[0]
            floor = np.mean((X @ ols - y) ** 2)

            plt.figure(figsize=(7, 4))
            plt.plot(losses, color="#0284c7", lw=1.6, label="GD loss")
            plt.axhline(floor, ls="--", c="#94a3b8", lw=1, label=f"闭式解底线 {floor:.6f}")
            plt.axvline(times, ls=":", c="#f97316", lw=1, label=f"收敛 {times} 步")
            plt.yscale("log")
            plt.xlabel("迭代步数");
            plt.ylabel("loss (MSE)")
            plt.title("一元线性回归 · loss 下降曲线")
            plt.grid(alpha=.3, which="both");
            plt.legend()
            plt.savefig("loss_curve.png", dpi=120)
            plt.show()

        case 2:
            numbers=1000
            """数据集构建"""
            np.random.seed(0)
            x=np.random.uniform(0, 100, numbers)
            w_r=np.random.randint(0, 5)
            b_r=np.random.randint(0, 5)
            # bios=np.random.uniform(0,1e-2,numbers)
            bios=np.random.randn(numbers) * 0.5

            """标准化"""
            x_mean,x_std=x.mean(),x.std()
            x_s=(x-x_mean)/x_std

            """训练集"""
            y=x_s*w_r+b_r+bios

            """训练过程"""
            losses1=[]
            lr=0.05
            w=0
            b=0
            times1=None
            for i in range(1000):
                l=loss(w, x_s, b, y)
                dw,db =grads(w, x_s, b, y)
                w-=dw*lr
                b-=db*lr
                losses1.append(l)
                if np.linalg.norm([dw, db]) < 1e-12:
                    times1=i
                    break
            losses2=[]
            lr=0.05
            w=0
            b=0
            times2=None
            for i in range(1000):
                l=loss(w,x_s,b,y)
                dw,db=gradsc(w,x_s,b,y)
                w-=dw*lr
                b-=db*lr
                losses2.append(l)
                if np.linalg.norm([dw,db])<1e-12:
                    times2=i
                    break
            print("公式次数，公式loss，差分次数，差分loss")
            print([times1,losses1[-1],times2,losses2[-1]])
        case 3:
            numbers=1000
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
            y=x_s*w_r+b_r+bios

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
                if np.linalg.norm([dw,db]) < 1e-12:
                    times=i
                    break
            atl=[]
            for i in (1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7):
                atl.append(Atl(i,np.allclose([w,b],[w_r,b_r],atol=i)))
            with open("对拍精度.csv","w",encoding="UTF-8-sig",newline='') as f:
                w=csv.writer(f)
                w.writerow(["数量级","是否对拍成功"])
                w.writerows(line.to_row() for line in atl)
        case 4:
            numbers = 1000
            """数据集构建"""
            np.random.seed(0)
            x = np.random.uniform(0, 100, numbers)
            w_r = np.random.randint(0, 5)
            b_r = np.random.randint(0, 5)
            # bios=np.random.uniform(0,1e-2,numbers)
            bios = np.random.randn(numbers) * 0.5

            """标准化"""
            x_mean, x_std = x.mean(), x.std()
            x_s = (x - x_mean) / x_std

            """训练集"""
            y = x_s * w_r + b_r + bios

            w=0
            b=0
            t=[]
            for n in range(100, 10001, 100):
                t0=time.perf_counter()
                for _ in range(n):
                    grads(w, x_s, b, y)
                t1=time.perf_counter()
                func=t1-t0
                t0=time.perf_counter()
                for _ in range(n):
                    gradsc(w , x_s, b, y)
                t1 = time.perf_counter()
                diff = t1 - t0
                t.append(timeg(n, func, diff))

            with open("time consume.csv", 'w', encoding="UTF-8-sig", newline='') as f:
                w = csv.writer(f)
                w.writerow(["times", "function", "difference"])
                w.writerows(line.to_row() for line in t)

        case 5:
            numbers = 1000
            """数据集构建"""
            np.random.seed(0)
            x = np.random.uniform(0, 100, numbers)
            w_r = np.random.randint(0, 5)
            b_r = np.random.randint(0, 5)
            # bios=np.random.uniform(0,1e-2,numbers)
            bios = np.random.randn(numbers) * 0.5

            """标准化"""
            x_mean, x_std = x.mean(), x.std()
            x_s = (x - x_mean) / x_std

            """训练集"""
            y = x_s * w_r + b_r + bios

            """三次loss的误差分析，以w为例"""
            w = 0
            b = 0
            hs = 10.0 ** -np.arange(1, 17)  # 1e-1 ... 1e-16
            errs = []
            for h in hs:
                dn,dm = gradcthree(w, x_s, b, y, h)
                dw,db = gradthree(w, x_s, b, y)
                errs.append(abs(dn - dw) / abs(dw))
            errs = np.array(errs)

            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 中文不乱码
            plt.rcParams["axes.unicode_minus"] = False

            plt.figure(figsize=(7, 4))
            plt.loglog(hs, errs, "o-", color="#0284c7", lw=1.6, label="实测相对误差")
            plt.gca().invert_xaxis()  # ← 让 h 从 1e-1 走到 1e-16，谷才在中间
            plt.axvline(1e-5, ls="--", c="#94a3b8", lw=1, label="理论最优 h ≈ ε^(1/3)")
            plt.xlabel("步长 h");
            plt.ylabel("相对误差 |diff-formula| / |formula|")
            plt.title("差分求导的 U 形谷 · 三次 loss")
            plt.grid(alpha=.3, which="both")  # both 才会画出次刻度网格
            plt.legend()
            plt.savefig("h_scan.png", dpi=120)  # 先存盘，再 show
            plt.show()

        case 6:
            data=[]
            for numbers in (10,20,50,100,1000,10000,50000,100000):
                """数据集构建"""
                np.random.seed(0)
                x = np.random.uniform(0, 100, numbers)
                w_r = np.random.randint(0, 5)
                b_r = np.random.randint(0, 5)
                # bios=np.random.uniform(0,1e-2,numbers)
                bios = np.random.randn(numbers) * 0.5

                """标准化"""
                x_mean, x_std = x.mean(), x.std()
                x_s = (x - x_mean) / x_std

                """训练集"""
                y = x_s * w_r + b_r + bios

                """训练过程"""
                losses = []
                lr = 0.05
                w = 0
                b = 0
                times = None
                for i in range(1000):
                    l = loss(w, x_s, b, y)
                    dw, db = grads(w, x_s, b, y)
                    w -= dw * lr
                    b -= db * lr
                    losses.append(l)
                    if np.linalg.norm([dw, db]) < 1e-12:
                        times = i
                        break

                """验证泛化能力"""
                xte = np.random.uniform(0, 100, 100)
                bios = np.random.randn(100) * 0.5
                xte_s = (xte - x_mean) / x_std
                yte = w_r * xte_s + b_r + bios
                val = np.mean(loss(w, xte_s, b, yte))
                data.append(Data(times, numbers, losses[-1], val, val-losses[-1]))

            """写入csv"""
            with open("泛化能力比较.csv", 'w', encoding="UTF-8-sig", newline='') as f:
                w = csv.writer(f)
                w.writerow(["收敛步数","数据量","训练末loss","测试集loss","两者差距"])
                w.writerows(line.to_row() for line in data)


    order=int(input("----输入一个数----"))


