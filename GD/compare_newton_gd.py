import math
import sys
import matplotlib.pyplot as plt
import numpy as np

arr=[]
iterion=[]
def newtons(f,df,x0,e,array,itr):
    xn = float(x0)
    e_tmp = e+1
    loop = 1
    while e_tmp>e:
        #print ('loop'+str(loop))
        k = df(xn)
        xm = f(xn)
        #print ('xn='+str(xn)+',k='+str(k)+',y='+str(xm))
        q = xm/k
        xn = xn-q
        array.append(xm+1);
        itr.append(loop)
        e_tmp = abs(0-f(xn))
        #print ('new xn='+str(xn)+',e='+str(e_tmp)+',q='+str(q))
        loop=loop+1
    return xn  

def GD(f,df,x_start,epochs, lr):

    xs = np.zeros(epochs+1)
    x = x_start
    xs[0] = x
    for i in range(epochs):
        dx = df(x)
        v = - dx * lr
        x += v
        xs[i+1] = x
    return xs
    
def f(x):
    return x**2+2*x
    #x2+2x
def df(x):
    return 2*x+2


 
y = newtons(f,df,3,0.01,arr,iterion)
y1= f(GD(f,df,3,140,0.01))+1
plt.plot(iterion,arr,color='red',label='newton')
plt.plot(np.arange(1,142,1),y1,color='green',label='bgd')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.show()
