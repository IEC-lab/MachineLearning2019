import matplotlib.pyplot as plt
from scipy.stats import beta
import numpy as np
import math


def plotMLE(X,Theta):
    X_s = sum(X)
    n = len(X)
    P = list(map(lambda theta: math.log(theta)*X_s+math.log(1.0 - theta)*(n - X_s),Theta))
    max_p = max(P)
    max_theta = Theta[P.index(max_p)]

    plt.plot(Theta,P,label="%d data points (MLE)"% n)
    plt.plot(max_theta,max_p,marker='o')
    plt.title(r"log-likelihood function $l(\theta)$ vs $\theta$")
    plt.legend(loc="lower left")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$l(\theta)$")
    plt.show()
def plotMAP(X,Theta,a,b):
    X_s = sum(X)
    n = len(X)
    P = list(map(lambda theta:math.log(theta)*X_s+math.log(1.0-theta)*(n-X_s)+math.log(beta.pdf(theta,a,b)), Theta))
    #P = list(map(lambda theta: math.log(theta) * (X_s + a - 1) + math.log(1.0 - theta) * (n - X_s + b - 1), Theta))
    max_p = max(P)
    max_theta = Theta[P.index(max_p)]

    plt.plot(Theta,P,label="%d data points (MAP)" % n)
    plt.plot(max_theta,max_p,marker='o')
    plt.title(r"log-posterior function $l(\theta)$ vs $\theta$")
    plt.legend(loc="lower left")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$l(\theta)$")
    plt.show()


def main():
    Theta = np.linspace(0.001,1.0,100,endpoint=False)
    plotMLE(x1,Theta)
    plotMLE(x2,Theta)
    plotMLE(x3,Theta)
    plt.savefig("MLE")
    plotMAP(x1,Theta,1,2)#Theta服从B(1,2)
    plotMAP(x2,Theta,1,2)
    plotMAP(x3,Theta,1,2)
    plt.savefig("MAP")

    return 0

# input
x1 = [0,1,1,1,0]
x2 = [0,1,1,1,0,0,0,0,0,1]
x3 = [0,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,1]

main()
