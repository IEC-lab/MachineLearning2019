'''
GDA-高斯判别分析
依赖库:
    python 3
    numpy
    matplotlib
尝试:
    pip install numpy
    pip install matplotlib
'''
import matplotlib.pyplot as plt
from numpy import *

#使用numpy.random随机生成两组高斯分布的聚类
mean0 = [2,3]
cov = mat([[1,0],[0,2]])
x0 = random.multivariate_normal(mean0,cov,100).T   #参数分别为均值，协方差矩阵，维度
y0 = zeros(shape(x0)[1])   #第一类标签为0

mean1 = [7,8]
cov = mat([[1,0],[0,2]])
x1 = random.multivariate_normal(mean1,cov,60).T
y1 = ones(shape(x1)[1])   #第二类标签为1

x = array([concatenate((x0[0],x1[0])),concatenate((x0[1],x1[1]))])
y = array([concatenate((y0,y1))])
m = shape(x)[1]

#计算参数:\phi,\u0,\u1,\Sigma
phi = (1.0/m)*len(y1)
u0 = mean(x0,axis=1)  
u1 = mean(x1,axis=1)

xplot0 = x0
xplot1 = x1   #保留原始数据做画图用
x0 = x0.T
x1 = x1.T
x = x.T

x0_sub_u0 = x0-u0
x1_sub_u1 = x1-u1

x_sub_u = concatenate([x0_sub_u0,x1_sub_u1])

x_sub_u = mat(x_sub_u)

sigma = (1.0/m)*(x_sub_u.T*x_sub_u)

#绘制判别边界,使用聚类均值中心u0和u1的垂直平分线
midPoint = [(u0[0]+u1[0])/2.0,(u0[1]+u1[1])/2.0]

k = (u1[1]-u0[1])/(u1[0]-u0[0])

x = range(-2,11)
y = [(-1.0/k)*(i-midPoint[0])+midPoint[1] for i in x]

#绘制两个高斯分布的轮廓
def gaussian_2d(x, y, x0, y0, sigmaMatrix):
    return exp(-0.5*((x-x0)**2+0.5*(y-y0)**2))

delta = 0.025
xgrid0 = arange(-2, 6, delta)
ygrid0 = arange(-2, 6, delta)
xgrid1 = arange(3,11,delta)
ygrid1 = arange(3,11,delta)
X0, Y0 = meshgrid(xgrid0, ygrid0)   #生成网格
X1, Y1 = meshgrid(xgrid1,ygrid1)
Z0 = gaussian_2d(X0,Y0,2,3,cov)
Z1 = gaussian_2d(X1,Y1,7,8,cov)

#绘制图表并添加注释
plt.figure(1)
plt.clf()
plt.plot(xplot0[0],xplot0[1],'ko')
plt.plot(xplot1[0],xplot1[1],'gs')
plt.plot(u0[0],u0[1],'rx',markersize=20)
plt.plot(u1[0],u1[1],'y*',markersize=20)
plt.plot(x,y)
CS0=plt.contour(X0, Y0, Z0)
plt.clabel(CS0, inline=1, fontsize=10)
CS1=plt.contour(X1,Y1,Z1)
plt.clabel(CS1, inline=1, fontsize=10)
plt.title("Gaussian discriminat analysis")
plt.xlabel('Feature Dimension (0)')
plt.ylabel('Feature Dimension (1)')
plt.show(1)