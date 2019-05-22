import sys
import matplotlib.pyplot as plt
import numpy as np
import random

sgd_loss=[]
sgd_iterion=[]
bgd_loss=[]
bgd_iterion=[]
mbgd_loss=[]
mbgd_iterion=[]

def gen_line_data(sample_num=100):
    """
    y = 3*x1 + 4*x2
    :return:
    """
    x1 = np.linspace(0, 9, sample_num)
    x2 = np.linspace(4, 13, sample_num)
    x = np.concatenate(([x1], [x2]), axis=0).T
    y = np.dot(x, np.array([3, 4]).T)  # y 列向量
    return x, y

def sgd(samples, y, step_size=0.01, max_iter_count=200):
    """
    随机梯度下降法
    :param samples: 样本
    :param y: 结果value
    :param step_size: 每一接迭代的步长
    :param max_iter_count: 最大的迭代次数
    :param batch_size: 随机选取的相对于总样本的大小
    :return:
    """
    sample_num, dim = samples.shape
    y = y.flatten()
    w = np.ones((dim,), dtype=np.float32)
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim,), dtype=np.float32)
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            for j in range(dim):
                error[j] += (y[i] - predict_y) * samples[i][j]
                w[j] += step_size * error[j] / sample_num

        # for j in range(dim):
        #     w[j] += step_size * error[j] / sample_num

        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
            loss += error
        sgd_loss.append(loss)
        sgd_iterion.append(iter_count)
        #print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w

def bgd(samples, y, step_size=0.01, max_iter_count=200):

    sample_num, dim = samples.shape
    y = y.flatten()
    w = np.ones((dim,), dtype=np.float32)
    # batch_size = np.ceil(sample_num * batch_size)
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim,), dtype=np.float32)

        # batch_samples, batch_y = select_random_samples(samples, y,
        # batch_size)

        #index = random.sample(range(sample_num),
                              #int(np.ceil(sample_num * batch_size)))
        #batch_samples = samples[index]
        #batch_y = y[index]
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            for j in range(dim):
                error[j] += (y[i] - predict_y) * samples[i][j]
        for j in range(dim):
            w[j] += step_size * error[j] / sample_num
        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
            loss += error
        bgd_loss.append(loss)
        bgd_iterion.append(iter_count)
        #print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w

def mbgd(samples, y, step_size=0.01, max_iter_count=200, batch_size=0.2):
    """
    MBGD（Mini-batch gradient descent）小批量梯度下降：每次迭代使用b组样本
    :param samples:
    :param y:
    :param step_size:
    :param max_iter_count:
    :param batch_size:
    :return:
    """
    sample_num, dim = samples.shape
    y = y.flatten()
    w = np.ones((dim,), dtype=np.float32)
    # batch_size = np.ceil(sample_num * batch_size)
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim,), dtype=np.float32)

        # batch_samples, batch_y = select_random_samples(samples, y,
        # batch_size)

        index = random.sample(range(sample_num),
                              int(np.ceil(sample_num * batch_size)))
        batch_samples = samples[index]
        batch_y = y[index]

        for i in range(len(batch_samples)):
            predict_y = np.dot(w.T, batch_samples[i])
            for j in range(dim):
                error[j] += (batch_y[i] - predict_y) * batch_samples[i][j]

        for j in range(dim):
            w[j] += step_size * error[j] / sample_num

        for i in range(sample_num):
            predict_y = np.dot(w.T, samples[i])
            error = (1 / (sample_num * dim)) * np.power((predict_y - y[i]), 2)
            loss += error
        mbgd_loss.append(loss)
        mbgd_iterion.append(iter_count)
        #print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w

if __name__ == '__main__':
    samples, y = gen_line_data()
    w = bgd(samples, y)
    w1= sgd(samples, y)
    w2= mbgd(samples,y)
    #print(w)  # 会很接近[3, 4]
    x=np.arange(0,len(bgd_iterion),1)
    x1=np.arange(0,len(sgd_iterion),1)
    x2=np.arange(0,len(mbgd_iterion),1)
    plt.plot(x,bgd_loss,color='red',label='bgd')
    plt.plot(x1,sgd_loss,color='blue',label='sgd')
    plt.plot(x2,mbgd_loss,color='green',label='mbgd')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
