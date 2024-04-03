#Import the libraries that we are going to need to carry out the analysis:
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
            prices = prices[prices[ticker] < 1e6]
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

    # Tính toán lợi suất hàng năm (returns) và độ biến động (volatility)
    returns = pd.DataFrame()
    returns['Returns'] = prices_df.pct_change(fill_method=None).mean() * 252
    returns['Volatility'] = prices_df.pct_change(fill_method=None).std() * sqrt(252)

    # In ra kết quả hoặc thực hiện các thao tác tiếp theo cần thiết
    print(returns)

# Format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T
X = data
distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
fig = plt.figure(figsize=(15, 5))

plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.show()
# Computing K-Means with K = 4 (4 clusters)
centroids,_ = kmeans(data,5)

# Assign each sample to a cluster
idx,_ = vq(data,centroids)

# Create a dataframe with the tickers and the clusters that's belong to
details = [(name,cluster) for name, cluster in zip(returns.index,idx)]
details_df = pd.DataFrame(details)

# Rename columns
details_df.columns = ['Ticker','Cluster']

# Create another dataframe with the tickers and data from each stock
clusters_df = returns.reset_index()

# Bring the clusters information from the dataframe 'details_df'
clusters_df['Cluster'] = details_df['Cluster']

# Rename columns
clusters_df.columns = ['Ticker', 'Returns', 'Volatility', 'Cluster']

# Plot the clusters created using Plotly
fig = px.scatter(clusters_df, x="Returns", y="Volatility", color="Cluster", hover_data=["Ticker"])
fig.update(layout_coloraxis_showscale=False)
fig.show()