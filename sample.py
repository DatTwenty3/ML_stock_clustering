import os
import numpy as np
import pandas as pd
from pylab import plot,show
from matplotlib import pyplot as plt
import plotly.express as px
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA

#Tạo các Centroid theo phương random
k  = 3
def select_initial_centers(X, k):
    centers = []
    centers.append(X[np.random.choice(X.shape[0])])
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x-c) for c in centers]) for x in X])
        probabilities = distances / distances.sum()
        cum_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cum_probabilities):
            if r < p:
                i = j
                break
        centers.append(X[i])
    return centers

#Các hàm sử dụng để update các Centroid
def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []

        curr_x = X[idx]

        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

# Implementing the M-Step
def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center

            clusters[i]['points'] = []
    return clusters

def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

# Đường dẫn đến thư mục chứa các tệp dữ liệu cổ phiếu
folder_path = "Stock_for_clustering(S&P500)"

# Tạo một danh sách để lưu trữ dữ liệu giá cổ phiếu từ các tệp
prices_list = []

# Lặp qua tất cả các tệp trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        # Lấy tên mã cổ phiếu từ tên tệp (loại bỏ phần mở rộng .dat)
        ticker = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)
        try:
            # Đọc chỉ 252 dòng từ tệp dữ liệu
            prices = pd.read_csv(file_path, header=None, names=[ticker], nrows=252)
            # Loại bỏ các giá trị có hơn 6 chữ số
            #prices = prices[prices[ticker] < 1e6]
            prices_list.append(prices)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

# Kiểm tra xem có dữ liệu nào được đọc không
if len(prices_list) == 0:
    print("Không có dữ liệu nào được đọc từ các tệp.")
else:
    # Nếu có ít nhất một dữ liệu được đọc, tiếp tục xử lý
    # Ghép các DataFrame chứa giá cổ phiếu của các mã thành một DataFrame lớn
    prices_df = pd.concat(prices_list, axis=1)
    prices_df.sort_index(inplace=True)
    print(prices_df)
    # Tính toán lợi suất hàng năm (returns) và độ biến động (volatility)
    returns = pd.DataFrame()
    returns['Returns'] = (prices_df.pct_change(fill_method=None)/100).mean() * 252
    returns['Volatility'] = (prices_df.pct_change(fill_method=None)/100).std() * sqrt(252)
    # Thực hiện gom cụm sử dụng dữ liệu returns
    X = returns.values

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1])
plt.show()

# Format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
x = data
distorsions = []
for n in range(2, 15):
    k_means = KMeans(n_clusters = n)
    k_means.fit(x)
    distorsions.append(k_means.inertia_)
#fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 15), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()

# Sử dụng hàm select_initial_centers để chọn các centroid ban đầu
centers = select_initial_centers(X, k)

# Khởi tạo clusters sử dụng các centroid được chọn
clusters = {}
for idx, center in enumerate(centers):
    cluster = {
        'center': center,
        'points': []
    }
    clusters[idx] = cluster

# In ra các centroid ban đầu
print("Initial Centroids:")
for idx, center in enumerate(centers):
    print(f"Centroid {idx+1}: {center}")

plt.scatter(X[:, 0], X[:, 1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', c='red')
plt.show()

clusters = assign_clusters(X,clusters)
clusters = update_clusters(X,clusters)
pred = pred_cluster(X,clusters)

plt.scatter(X[:,0],X[:,1],c = pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.grid(True)
plt.show()