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
            prices = pd.read_csv(file_path, header=None, names=[ticker], nrows=1000)
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
centers = select_initial_centers(prices_df, k)



