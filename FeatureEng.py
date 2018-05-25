import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import math

iris = load_iris()
n_samples, n_features = iris.data.shape
print(iris.keys())
data = iris.data
target = iris.target
target_names = iris.target_names
feature_names = iris.feature_names
df = pd.DataFrame(data, columns=feature_names)

#  ############  1.变量统计
#  1.1变量分布
df.plot()  # 线形图
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, fontsize=8, figsize=(8, 6))
df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, fontsize=8, figsize=(8, 6))
df.plot(kind='bar', subplots=True)  # 柱状图,每个变量分别作图
#  kind取值：'line','bar','barh','hist','box','kde','density','area','pie','scatter','hexbin'
df.hist()  # 柱状图
#  1.2变量相关性
#  散布矩阵
pd.plotting.scatter_matrix(df, alpha=1, figsize=(14, 8), ax=None, diagonal='kde', marker='.', density_kwds=None, hist_kwds=None, range_padding=0.05)
#  相关系数,热力图
cov = np.corrcoef(df.T)
img = plt.matshow(cov, cmap=plt.cm.winter)
plt.colorbar(img, ticks=[-1, 0, 1])
plt.xticks(np.arange(len(df.keys())), df.keys())
plt.yticks(np.arange(len(df.keys())), df.keys())
plt.show()
#  1.3类别散点图
def scatterDiagram(data, target, target_names, feature_names):
    n_samples, n_features = data.shape
    fig = plt.figure()
    n = math.floor(math.sqrt(n_features * (n_features - 1) / 2))+1
    for i in range(1, n_features+1):
        for j in range(i+1, n_features+1):
            x_index, y_index = i, j
            ax = fig.add_subplot(n, n, ((j-i)+(2*n_features-i)*(i-1)/2))
            cs = ax.scatter(data[:, x_index-1], data[:, y_index-1], s=5, c=target, marker='.')
            ax.set_xlabel(feature_names[x_index-1])
            ax.set_ylabel(feature_names[y_index-1])
    formatter = plt.FuncFormatter(lambda x, *args: target_names[int(x)])
    ticks_list = list(set(target))
    cbar = fig.colorbar(cs, ticks=ticks_list, format=formatter, orientation='horizontal',                        cax=plt.axes([0.05, 0.06, 0.3, 0.02]))
    fig.show()
    return()


#  ############  2.特征工程
#  2.2 无量纲化
#  2.1.1、标准化
from sklearn.preprocessing import StandardScaler
data1 = StandardScaler().fit_transform(df)
#  2.1.2、 区间缩放
from sklearn.preprocessing import MinMaxScaler
data2 = MinMaxScaler().fit_transform(df)
#  2.1.3、 归一化
from sklearn.preprocessing import Normalizer
data3 = Normalizer().fit_transform(df)
#  2.2 定量特征二值化
from sklearn.preprocessing import Binarizer
data4 = Binarizer(threshold=3).fit_transform(df)  #二值化，阈值设置为3，返回值为二值化后的数据
#  2.3 对定性特征哑编码
from sklearn.preprocessing import OneHotEncoder
data5 = OneHotEncoder().fit_transform(target.reshape((-1, 1))).toarray()
#  2.3 缺失值计算
#  2.3.1 缺失值填充
data.fillna(method='', inplace=True)
np.mean(), sum()
#  2.4 数据变换
#  2.4.1 多项式变换
from sklearn.preprocessing import PolynomialFeatures
data6 = PolynomialFeatures().fit_transform(df)
#  2.4.2 对数变换
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
data7 = FunctionTransformer(log1p).fit_transform(df)


#  ############  3.变量选择
#  3.1 Filter
#  3.1.1 方差选择法
from sklearn.feature_selection import VarianceThreshold
data_f = VarianceThreshold(threshold=3).fit_transform(df)   # 参数threshold为方差的阈值
#  3.1.2 相关系数法
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
data_f = SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(data, target)
#  3.1.3 卡放检验法
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data_f = SelectKBest(chi2, k=2).fit_transform(data, target)
#  3.1.4 互信息法
from sklearn.feature_selection import SelectKBest
from minepy import MINE
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(data, target)

#  3.2Wrapper
#  3.2.1递归特征消除法
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#  递归特征消除法，返回特征选择后的数据
#  参数estimator为基模型
#  参数n_features_to_select为选择的特征个数
data_f = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(data, target)

#  3.3 Embedded
#  3.3.1 基于惩罚项的特征选择法
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
#  带L1惩罚项的逻辑回归作为基模型的特征选择
data_f = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(data, target)
# 带L1和L2惩罚项的逻辑回归作为基模型的特征选择
class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self
# 参数threshold为权值系数之差的阈值
data_f = SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(data, target)
#  3.3.2 基于树模型的特征选择法
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
#  GBDT作为基模型的特征选择
data_f = SelectFromModel(GradientBoostingClassifier()).fit_transform(data, target)


#  ############  3.降维
#  4.1 主成分分析 （PCA）
from sklearn.decomposition import PCA
#  主成分分析法，返回降维后的数据
#  参数n_components为主成分数目
data_f = PCA(n_components=2).fit_transform(data)
#  4.2 现行判别分析法 （LDA）
from sklearn.lad import LDA
#  线性判别分析法，返回降维后的数据
#  参数n_components为降维后的维数
data_f = LDA(n_components=2).fit_transform(data, target)
