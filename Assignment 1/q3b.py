import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
df = pd.read_excel('x_train.xlsx')
data = df.to_numpy()

N = data.shape[0]          
Y = pd.read_csv('y_train.csv')
label = np.squeeze(Y.to_numpy())

data = data[:, 0:-2]
sc = StandardScaler()
data = sc.fit_transform(data)

p_c_a = PCA(n_components = 10)
data = p_c_a.fit_transform(data)

# Total labels
total_labels = 6  

# Total features
feature = 10         

# Computing Mean Vectors and Covariance matrix
matrix_mean = np.zeros(shape = [total_labels, feature])
covariance_matrix = np.zeros(shape = [total_labels, feature, feature])

for i in range(0, total_labels):
    matrix_mean[i, :] = np.mean(data[(label == i + 1), :], axis = 0)
    covariance_matrix[i, :, :] = np.cov(data[(label == i + 1), :], rowvar = False)
    covariance_matrix[i, :, :] += (0.00001) * ((np.trace(covariance_matrix[i,:,:]))/LA.matrix_rank(covariance_matrix[i,:,:])) * np.eye(10)

loss_matrix = np.ones(shape = [total_labels, total_labels]) - np.eye(total_labels)

# Computing class conditional PDF
px_given = np.zeros(shape = [total_labels, N])
for i in range(0, total_labels):
    px_given[i, :] = multivariate_normal.pdf(data, mean = matrix_mean[i, :], cov = covariance_matrix[i, :,:])

# Estimating class classprior
classprior = np.zeros(shape = [total_labels, 1])
for i in range(0, total_labels):
    classprior[i] = (np.size(label[np.where((label == i + 1))])) / N

# Computing Class Posteriors
px = np.matmul(np.transpose(classprior), px_given)
class_post = (px_given * (np.matlib.repmat(classprior, 1, N))) / np.matlib.repmat(px, total_labels, 1)

# Evaluating Expected Risk
exp_risk = np.matmul(loss_matrix, class_post)
dec = np.argmin(exp_risk, axis = 0)
print("Average Expected Risk:", np.sum(np.min(exp_risk, axis = 0)) / N)

# Estimating Confusion Matrix
conf_matrix = np.zeros(shape = [total_labels, total_labels])
for d in range(total_labels):
    for l in range(total_labels):
        conf_matrix[d, l] = (np.size(np.where((d == dec) & (l == label - 1)))) / np.size(np.where(label - 1 == l))

np.set_printoptions(suppress=True)
print(conf_matrix)

# Plotting Data Distribution Graph
fig = plt.figure()
ax = plt.axes(projection = "3d")
for i in range(1, total_labels + 1):
    ax.scatter(data[(label==i),1],data[(label==i),2],data[(label==i),3], label=i)
plt.xlabel('X3')
plt.ylabel('X1')
ax.set_zlabel('X2')
ax.legend()
plt.title('Data Distribution Graph')
plt.show()
