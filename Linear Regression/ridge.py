import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split

def load_data():
    diabetes = datasets.load_diabetes()
    return train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)

def test_ridge(*data):
    X_train, X_test, y_train, y_test = data
    ridgeRegression = linear_model.Ridge()
    ridgeRegression.fit(X_train, y_train)
    print("权重向量:%s, b的值为:%.2f" % (ridgeRegression.coef_, ridgeRegression.intercept_))
    print("损失函数的值:%.2f" % np.mean((ridgeRegression.predict(X_test) - y_test) ** 2))
    print("预测性能得分: %.2f" % ridgeRegression.score(X_test, y_test))

#测试不同的α值对预测性能的影响
def test_ridge_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        ridgeRegression = linear_model.Ridge(alpha=alpha)
        ridgeRegression.fit(X_train, y_train)
        scores.append(ridgeRegression.score(X_test, y_test))
    return alphas, scores

def show_plot(alphas, scores):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale("log")
    ax.set_title("Ridge")
    plt.show()

if __name__ == '__main__':
    #使用默认的alpha
    #获得数据集
    X_train, X_test, y_train, y_test = load_data()
    #进行训练并且预测结果
    test_ridge(X_train, X_test, y_train, y_test)

    #使用自己设置的alpha
    #X_train, X_test, y_train, y_test = load_data()
    alphas, scores = test_ridge_alpha(X_train, X_test, y_train, y_test)
    show_plot(alphas, scores)
