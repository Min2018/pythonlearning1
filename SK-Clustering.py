import numpy as np
from sklearn import cluster, datasets


'''
K-means clustering : k均值聚类
聚类中存在许多不同的聚类标准和相关算法，最简单的聚类算法就是K均值聚类算法
'''
k_means = cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                         tol=0.0001, precompute_distances='auto', verbose=0, random_state=None,
                         copy_x=True, n_jobs=1, algorithm='auto')
'''
参数：
n_clusters：分成的簇数（要生成的质心数）n_clusters：分成的簇数（要生成的质心数）
init：初始化质心的方法，有三个可选值：'k-means++'， 'random'，或者传递一个ndarray向量，默认为'k-means++'
        'k-means++' 用一种智能的方法选定初始质心从而能加速迭代过程的收敛，
        'random' 随机从训练数据中选取初始质心。
         如果传递的是一个ndarray，则应该形如 (n_clusters, n_features) 并给出初始质心
n_init:：用不同的质心初始化值运行算法的次数====>整型，默认值=10次，最终解是在inertia意义下选出的最优结果
max_iter：算法每次迭代的最大次数====>整型，默认值=300
tol：与inertia结合来确定收敛条件====> float型，默认值= 1e-4
precompute_distances：预计算距离，计算速度更快但占用更多内存 ====>类型：（auto，True，False）三个值可选，,默认值=“auto”
        ‘auto’：如果样本数乘以聚类数大于 12million 的话则不预计算距离‘’
        ‘True‘：总是预先计算距离。
        ‘False‘：永远不预先计算距离。
verbose:是否输出详细信息====>类型：整型，默认值=0
random_state： 用于初始化质心的生成器（generator），和初始化中心有关。
copy_x：是否对输入数据继续copy 操作====> 布尔型，默认值=True
        当我们precomputing distances时，将数据中心化会得到更准确的结果。   
        如果把此参数值设为True，则原始数据不会被改变。
        如果是False，则会直接在原始数据上做修改并在函数返回值时将其还原。
        但是在计算过程中由于有对数据均值的加减运算，所以数据返回后，原始数据和计算前可能会有细小差别。
n_jobs：使用进程的数量，与电脑的CPU有关====>类型：整型，默认值=1
        指定计算所用的进程数。内部原理是同时进行n_init指定次数的计算。
        若值为 -1，则用所有的CPU进行运算。
        若值为1，则不进行并行运算，这样的话方便调试。
        若值小于-1，则用到的CPU数为(n_cpus + 1 + n_jobs)。因此如果 n_jobs值为-2，则用到的CPU数为总CPU数减1
algorithm：k-means算法的种类====>（“auto”, “full” or “elkan”）三个值可选，默认值=‘auto’
        'full'采用的是经典EM-style算法的。
        'elkan'则在使用三角不等式时显得更为高效,但目前不支持稀疏数据。
        'auto'则在密集数据时选择“elkan”，在稀疏数据是选择“full”

'''
'''
属性：
cluster_centers_:找出聚类中心
Labels_:每个点的分类
inertia_ :每个点到其簇的质心的距离之和
'''









iris = datasets.load_iris()  # 加载数据集
x_iris = iris.data  # 数据集的数据
y_iris = iris.target  # 数据集的标签
k_means = cluster.KMeans(n_clusters=3)  # k_means分类器,参数n_clusters=3,划分成3类

print(k_means.fit(x_iris))  # 分类器直接对数据进行聚类
print(k_means.labels_[::10])  # 标签
print(y_iris[::10])



import scipy as sp
import matplotlib.pyplot as plt

try:
    face = sp.face(gray=True)
except AttributeError:
    from scipy import misc
    face = misc.face(gray=True)

plt.gray()
plt.imshow(face)
plt.show()  # 显示原图

X = face.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=5, n_init=1)  # 构造分类器，参数n_clusters是K值
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)  #按照labels的序号对values中的数进行选择。
face_compressed.shape = face.shape

plt.gray()
plt.imshow(face_compressed)
plt.show()  # 显示分类器操作过的图像





















len(face_compressed)

face_compressed.shape
X.shape




