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

# Sử dụng hàm select_initial_centers để chọn các centroid ban đầu
#centers = select_initial_centers(prices_df, k)


def find_nearest_row_by_median(fm_data, cvmax, mean):

    if fm_data is None :
        print("==data is None")
        return

    if cvmax is None :
        print("cvmax is None")
        return
    #print(f"[số cụm k={k}], cvmax:{cvmax}")

    if mean is None :
        print("==mean is None")
        return
    #print(f"[số cụm k={k}], mean:{mean}")

    min = 9999999999999999

    for index, row in fm_data.iterrows():
        if abs(row[cvmax] - mean) < min:
            result = row.to_numpy()
    #print("result")
    #print(result)
    return result

def find_centroid_by_al_daoud_method(data, k):


    # Bước 1: Tính phương sai của mỗi thuộc tính
    cvmax = data.var().idxmax()

    # Bước 2: Tìm thuộc tính có phương sai lớn nhất và sắp xếp dữ liệu theo thuộc tính này
    #cvmax_index = np.argmax(variances)
    #sorted_data = data[np.lexsort((-data[:, cvmax_index],))]

    print(f"[số cụm k={k}]  cvmax:{cvmax}")

    # Bước 3: Chia dữ liệu thành k tập con
    data.sort_values(by = cvmax)
    data_chunks = np.array_split(data[cvmax], k, axis=0)
    #print("==data_chunks:")
    #print(data_chunks)


    # Bước 4: Tìm giá trị trung vị của mỗi tập con
    mediansArr = []
    for i in range(k):

        medians = np.median(data_chunks[i], axis=0)
        print(f"medians Cụm {i + 1}: {medians}")
        mediansArr.append(medians)


    # Bước 5: Sử dụng giá trị trung vị làm centroid ban đầu
    l_centroids = []
    for i in range(k):
        median = mediansArr[i]
        row = find_nearest_row_by_median(prices_df, cvmax, median)
        l_centroids.append(row)

        #print("median row")

    print(f"[số cụm k={k}] centroids lenght: {l_centroids.__len__()}.")
    #print(l_centroids)
    return l_centroids

# Ví dụ sử dụng

#print("===centroids===")
#centroids = find_centroid_by_al_daoud_clustering(prices_df, k)
#print("===centroids===")
#print(centroids)




# Create the plot
plt.figure(figsize=(8, 6))

data = np.array(prices_df)

distorsions = []
for k in range(2, 15):

    print("")
    print("=============================================================")
    print("=============================================================")
    print(f"với số cụm k={k}")
    print("=============================================================")
    print("=============================================================")
    centroids = find_centroid_by_al_daoud_method(prices_df, k)
    copied_centroids = np.copy(centroids)
    k_means = KMeans(n_clusters=k, init = centroids)

    k_means.fit(data)
    distorsions.append(k_means.inertia_)
#fig = plt.figure(figsize=(15, 5))

plt.plot(range(2, 15), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()