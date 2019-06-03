# -*- coding: utf-8 -*-
"""
svm test
"""

from SVM import get_data_set
from SVM import trainSVM
from SVM import draw_train
from SVM import draw_test

# 获取数据
x_train, y_train, x_test, y_test = get_data_set()
SVM = trainSVM(x_train, y_train)        # 训练
draw_train(SVM)                         # 画图显示训练结果
draw_test(SVM, x_test, y_test)          # 画图显示测试结果
