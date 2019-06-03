# -*- coding: utf-8 -*-
"""
test soft SVM with different C
"""

from SVM import get_data_set
from SVM import trainSVM
from SVM import draw_train
from SVM import draw_test

# 获取数据
x_train, y_train, x_test, y_test = get_data_set()
        
# C越大，分类器越严格，但是容易产生过拟合
# C越小，分类器比较松弛，允许一定范围内出现错误
C = [0.01, 0.1, 1, 10, 100]

for c in C:
    SVM = trainSVM(x_train, y_train, C = c)
    draw_train(SVM)
    draw_test(SVM, x_test, y_test)