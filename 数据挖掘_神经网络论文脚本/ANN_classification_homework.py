# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:24:41 2021

@author: 12932
"""
###

"""
一个样本有64个特征（0~63列），一个类标记（64列），64特征指8*8手写数字图片中64个像素点颜色值，
取值范围为0~16，类标记是图片的数字值0~9。
训练集3823样本，测试集1797样本。

"""
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import radviz
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
os.getcwdb()
#trainning set
data_train = np.genfromtxt('optdigits.tra',delimiter=',',dtype=float)
X_train,y_train = data_train[:,:-1],data_train[:, -1]#
#print(X_train)  特征
#print(y_train)  类标记

#test set
data_test = np.genfromtxt('optdigits.tes',delimiter=',',dtype=float)
X_test,y_test = data_test[:,:-1],data_test[:, -1]

ss=StandardScaler()
ss.fit(X_train)   #对训练集拟合
X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)##训练集测试集标准化

###类别编码，如（1,0,0,0,0,0,0,0,0,0,）表示数字为1
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train_bin = lb.fit_transform(y_train)

"""
创建了三层神经网络，输入层64节点，输出层10节点，自定义隐藏层100节点
eta随机梯度下降算法的学习率为0.3
max_iter迭代次数为500，判断收敛的误差阈值为0.00001

"""
from ann_classification import ANNClassifier
clf = ANNClassifier(hidden_layer_sizes=(100,),eta=0.3,max_iter=500,tol=0.00001)
clf.train(X_train_std,y_train_bin)
y_pred_bin = clf.predict(X_test_std)
from sklearn.metrics import accuracy_score
y_pred = lb.inverse_transform(y_pred_bin)
accuracy = accuracy_score(y_test,y_pred)

####change eta,eta=0.25
clf = ANNClassifier(hidden_layer_sizes=(100,),eta=0.25,max_iter=500,tol=0.00001)
clf.train(X_train_std,y_train_bin)
y_pred_bin = clf.predict(X_test_std)
y_pred = lb.inverse_transform(y_pred_bin)
accuracy_eta = accuracy_score(y_test,y_pred)
print(accuracy_eta)


####change hidden_layer_sizes,hidden_layer_sizes=150
clf = ANNClassifier(hidden_layer_sizes=(150,),eta=0.3,max_iter=500,tol=0.00001)
clf.train(X_train_std,y_train_bin)
y_pred_bin = clf.predict(X_test_std)
y_pred = lb.inverse_transform(y_pred_bin)
accuracy_hls = accuracy_score(y_test,y_pred)
print(accuracy_hls)


####change max_iter,max_iter=700
clf = ANNClassifier(hidden_layer_sizes=(100,),eta=0.3,max_iter=700,tol=0.00001)
clf.train(X_train_std,y_train_bin)
y_pred_bin = clf.predict(X_test_std)
y_pred = lb.inverse_transform(y_pred_bin)
accuracy_iter = accuracy_score(y_test,y_pred)
print(accuracy_iter)

####change tol,tol=0.000001
clf = ANNClassifier(hidden_layer_sizes=(150,),eta=0.3,max_iter=500,tol=0.000001)
clf.train(X_train_std,y_train_bin)
y_pred_bin = clf.predict(X_test_std)
y_pred = lb.inverse_transform(y_pred_bin)
accuracy_hls = accuracy_score(y_test,y_pred)
print(accuracy_hls)


