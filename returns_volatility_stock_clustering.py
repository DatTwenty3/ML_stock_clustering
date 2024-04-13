import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
from sklearn.cluster import KMeans

#Tạo các Centroid theo phương random
k  = 4
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
            prices = pd.read_csv(file_path, header=None, names=[ticker])#, nrows=252)
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
    print("Returns:\n")
    print((prices_df.pct_change(fill_method=None)).mean())
    print("Volatility:\n")
    print((prices_df.pct_change(fill_method=None)).std())
    # Tính toán lợi suất hàng năm (returns) và độ biến động (volatility)
    returns = pd.DataFrame()
    returns['Returns'] = (prices_df.pct_change(fill_method=None)).mean() #* 252
    returns['Volatility'] = (prices_df.pct_change(fill_method=None)).std() #* sqrt(252)
    # Thực hiện gom cụm sử dụng dữ liệu returns
    X = returns.values

fig = plt.figure(figsize=(10, 6))  # Tạo một hình với kích thước 10x6

# Vẽ điểm dữ liệu
plt.scatter(X[:, 0], X[:, 1])

# In tên của từng mã cổ phiếu tương ứng
for i, ticker in enumerate(returns.index):
    plt.text(X[i, 0], X[i, 1], ticker, fontsize=8, ha='right', va='bottom')

plt.xlabel('Lợi suất hàng năm')
plt.ylabel('Độ biến động')
plt.title('Biểu đồ "Lợi suất hằng năm" và "Độ biến động" của các mã cổ phiếu')
plt.grid(True)
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
plt.xlabel('Số lượng các cụm')
plt.ylabel('Độ biến dạng')
plt.title('Đường cong Elbow trước khi loại bỏ mã cổ phiếu RX và BXP')
plt.show()


# Loại bỏ mã cổ phiếu RX và BXP từ ma trận X và returns
X = X[~np.isin(returns.index, ['RX', 'BXP'])]
returns = returns[~returns.index.isin(['RX', 'BXP'])]

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
plt.xlabel('Số lượng các cụm')
plt.ylabel('Độ biến dạng')
plt.title('Đường cong Elbow sau khi loại bỏ mã cổ phiếu RX và BXP')
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

# In tên của từng mã cổ phiếu tương ứng
for i, ticker in enumerate(returns.index):
    plt.text(X[i, 0], X[i, 1], ticker, fontsize=8, ha='right', va='bottom')
plt.scatter(X[:, 0], X[:, 1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', c='red')
plt.xlabel('Lợi suất hàng năm')
plt.ylabel('Độ biến động')
plt.title('Biểu đồ thể hiện các điểm trọng tâm được  ngẫu nhiên ban đầu')
plt.show()

clusters = assign_clusters(X,clusters)
clusters = update_clusters(X,clusters)
pred = pred_cluster(X,clusters)

# Gán nhãn cụm cho mỗi quan sát trong tập dữ liệu ban đầu
returns['Cluster'] = pred

# Tạo một cấu trúc dữ liệu để lưu trữ mã cổ phiếu cho mỗi cụm
clusters_stocks = {i: [] for i in range(k)}

# Lặp qua từng quan sát và thêm mã cổ phiếu vào cụm tương ứng
for idx, row in returns.iterrows():
    cluster = row['Cluster']
    ticker = row.name
    clusters_stocks[cluster].append(ticker)

# In ra các mã cổ phiếu thuộc mỗi cụm
for cluster, stocks in clusters_stocks.items():
    print(f"Cluster {cluster + 1}: {', '.join(stocks)}")

# Tạo danh sách các ký hiệu để đại diện cho trung tâm của mỗi cụm
symbols = ['^', 's', 'D', 'x', '*', '+', 'p', 'h']

# In các mã cổ phiếu trong cụm tương ứng, màu của cụm và kí hiệu trung tâm
plt.figure(figsize=(10, 6))
for i, cluster_id in enumerate(clusters):
    center = clusters[cluster_id]['center']
    symbol = symbols[i % len(symbols)]  # Chọn ký hiệu từ danh sách
    plt.scatter(center[0], center[1], marker=symbol, c='red', label=f'Điểm trọng tâm của cụm {cluster_id + 1}')  # Trung tâm cụm
    # In các mã cổ phiếu trong cụm tương ứng
    for ticker in clusters_stocks[cluster_id]:
        point = returns.loc[ticker, ['Returns', 'Volatility']].values
        plt.text(point[0], point[1], ticker, fontsize=8)  # In tên mã cổ phiếu
plt.scatter(X[:, 0], X[:, 1], c=pred, alpha=0.5)  # Điểm dữ liệu với màu là cụm dự đoán
plt.xlabel('Lợi suất hàng năm')
plt.ylabel('Độ biến động')
plt.title('Biểu đồ thể hiện các mã cổ phiếu được phân cụm theo "Lợi suất hàng năm" và "Độ biến động" của các mã cổ phiếu')
plt.legend()
plt.grid(True)
plt.show()