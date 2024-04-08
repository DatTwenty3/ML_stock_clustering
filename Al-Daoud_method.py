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

# Đường dẫn đến thư mục chứa các tệp dữ liệu cổ phiếu
folder_path = "Stock_for_clustering(S&P500)"

# Tạo một danh sách để lưu trữ dữ liệu giá cổ phiếu từ các tệp

prices_df = df = pd.DataFrame()

# Lặp qua tất cả các tệp trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        # Lấy tên mã cổ phiếu từ tên tệp (loại bỏ phần mở rộng .dat)
        ticker = os.path.splitext(filename)[0]
        #print("==> reading file name filename:" + filename)
        prices_list = []
        file_path = os.path.join(folder_path, filename)
        try:
            # Đọc chỉ 252 dòng từ tệp dữ liệu
            prices = pd.read_csv(file_path, header=None, names=[ticker], nrows=200)
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
        prices_df[filename] = prices
        #print(prices_df.columns.values)


        # Tính toán lợi suất hàng năm (returns) và độ biến động (volatility)
        returns = pd.DataFrame()
        #returns['Returns'] = (prices_df.values.pct_change(fill_method=None)/100).mean() * 252

        # Thực hiện gom cụm sử dụng dữ liệu returns
        #X = returns.values

# Format the data as a numpy array to feed into the K-Means algorithm


#Tạo các Centroid theo phương pháp Al Daoud.
k  = 3
def select_initial_centers(X, k):
    centers = []

    #for column in prices_df.columns:
        #variance = df[column].var()
        #print(f"Phương sai của cột {column}: {variance}")
    column_with_max_variance = prices_df.var().idxmax()
    cvmax = column_with_max_variance
    print("cvmax:" + cvmax)


    return centers

# Sử dụng hàm select_initial_centers để chọn các centroid ban đầu
#centers = select_initial_centers(prices_df, k)


def find_min_row(data, cvmax, mean):

    min = 100000

    for index, row in data.iterrows():
        if abs(row[cvmax] - mean) < min:
            result = row.to_numpy()
    print("result")
    print(result)
    return result

def al_daoud_clustering(data, k):
    """
    Hàm thực hiện thuật toán phân cụm của M. B. Al-Daoud (2005)

    Tham số:
      data: Mảng NumPy chứa dữ liệu cần phân cụm (mỗi hàng là một điểm dữ liệu)
      k: Số lượng cụm mong muốn

    Trả về:
      Mảng NumPy chứa nhãn cụm cho mỗi điểm dữ liệu
    """

    # Bước 1: Tính phương sai của mỗi thuộc tính
    cvmax = data.var().idxmax()

    # Bước 2: Tìm thuộc tính có phương sai lớn nhất và sắp xếp dữ liệu theo thuộc tính này
    #cvmax_index = np.argmax(variances)
    #sorted_data = data[np.lexsort((-data[:, cvmax_index],))]

    print("==cvmax:" + cvmax)

    # Bước 3: Chia dữ liệu thành k tập con
    data.sort_values(by=cvmax)
    data_chunks = np.array_split(data[cvmax], k, axis=0)
    #print("==data_chunks:")
    #print(data_chunks)


    # Bước 4: Tìm giá trị trung vị của mỗi tập con
    mediansArr = []
    for i in range(k):
        print(f"medians Cụm {i + 1}:")
        medians = np.median(data_chunks[i], axis=0)
        print(medians)
        mediansArr.append(medians)


    # Bước 5: Sử dụng giá trị trung vị làm centroid ban đầu
    l_centroids = []
    for i in range(k):
        median = mediansArr[i]
        row = find_min_row(prices_df, cvmax, median)
        l_centroids.append(row)

        #print("median row")

    print(f"centroids len: {l_centroids.__len__()}")
    print(l_centroids)
    return l_centroids

# Ví dụ sử dụng

k = 3

centroids = al_daoud_clustering(prices_df, k)

print("Nhãn cụm:")
print(centroids)



model = KMeans(n_clusters=k, init=centroids, n_init=1)

# Phân cụm dữ liệu
model.fit(prices_df)
labels = model.labels_

print("==label")
print(model.labels_)

# Separate data points by cluster labels
data_clustered = []
for i in range(k):
    v_row = prices_df[labels == i].to_numpy()
    data_clustered.append(v_row)

print(centroids)
copied_centroids = np.copy(centroids)
#copied_cluster = np.copy(data_clustered)

# Create the plot
plt.figure(figsize=(8, 6))

#Plot the data points with different colors based on their cluster labels
for i in range(k):
    plt.scatter(data_clustered[i][:, 0], data_clustered[i][:, 1], label=f"Cluster {i+1}")

plt.scatter(copied_centroids[:, 0], copied_centroids[:, 1], marker='x', s=100, c='black', label='Centroids')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('K-Means Clustering Results (k = 3)')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()