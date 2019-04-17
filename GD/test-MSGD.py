import numpy as np
import random

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

def mbgd(samples, y, step_size=0.01, max_iter_count=10000, batch_size=0.2):
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

        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w

if __name__ == '__main__':
    samples, y = gen_line_data()
    w = mbgd(samples, y)
    print(w)  # 会很接近[3, 4]
