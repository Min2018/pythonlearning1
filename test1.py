from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import math
import pickle

iris = datasets.load_iris()  # 加载iris数据集
digits = datasets.load_digits()  # 加载digits数据集
print(iris)
print(digits)
print(iris.data)


digits = datasets.load_digits()  # 加载数据集

n_samples = len(digits.images)  # 样本的数量
data = digits.images.reshape((n_samples, -1))

classifier = svm.SVC(gamma=0.001)  # svm预测器
classifier.fit(data[:math.ceil(n_samples / 2)], digits.target[:math.ceil(n_samples / 2)])  # 使用数据集的一半进行训练数据

expected = digits.target[math.ceil(n_samples / 2):]
predicted = classifier.predict(data[math.ceil(n_samples / 2):])  # 预测剩余的数据

clf = svm.SVC()  # 构造预测器
iris = datasets.load_iris()  # 加载数据集
X, y = iris.data, iris.target  # 数据的样本数和结果数
clf.fit(X, y)  # 训练数据

s = pickle.dumps(clf)  # 保存训练模型
clf2 = pickle.loads(s)  # 加载训练模型
print(clf2.predict(X[0:1]))  # 应用训练模型






