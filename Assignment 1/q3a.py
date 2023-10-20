import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from numpy import linalg as LA

# Import Dataset
df = pd.read_csv('winequality-white.csv')
data = df.to_numpy()

# Number of Samples
samples = data.shape[0]  

# Seperate column containing class labels
label = data[:, 11] 

# Feature set
data = data[:, 0:11]   

# Number of classes
total_labels = 11         

# Number of features
features = 11         
matrix_mean = np.zeros(shape = [total_labels, features])
covariance_matrix = np.zeros(shape = [total_labels, features, features])

# Computing Mean Vectors and Covariance Matrix
for i in range(0, total_labels):
    matrix_mean[i, :] = np.mean(data[(label == i)], axis = 0)
    if (i not in label):
        covariance_matrix[i, :, :] = np.eye(features)
    else:
        covariance_matrix[i, :, :] = np.cov(data[(label == i), :], rowvar = False)
        covariance_matrix[i, :, :] += (0.000000005) * ((np.trace(covariance_matrix[i, :, :]))/LA.matrix_rank(covariance_matrix[i, :, :])) * np.eye(features)

# Assign 0-1 loss matrix
loss_matrix = np.ones(shape = [total_labels, total_labels]) - np.eye(total_labels)

# Computing class conditional PDF
px_given = np.zeros(shape = [total_labels, samples])
for i in range(0, total_labels):
    if i in label:
        px_given[i, :] = multivariate_normal.pdf(data, mean = matrix_mean[i, :], cov = covariance_matrix[i, :,:])

# Estimating class classprior
classprior = np.zeros(shape = [11, 1])
for i in range(0, total_labels):
    classprior[i] = (np.size(label[np.where((label == i))])) / samples

# Computing Class Posteriors
px = np.matmul(np.transpose(classprior), px_given)
class_post = (px_given * (np.matlib.repmat(classprior, 1, samples))) / np.matlib.repmat(px, total_labels, 1)

# Evaluating Expected Risk
exp_risk = np.matmul(loss_matrix, class_post)
dec = np.argmin(exp_risk, axis = 0)
print("Average Expected Risk:", np.sum(np.min(exp_risk, axis = 0)) / samples)

# Estimating Confusion Matrix
conf_matrix = np.zeros(shape = [total_labels, total_labels])
for d in range(total_labels):
    for l in range(total_labels):
        if l in label and d in label:
            conf_matrix[d, l] = (np.size(np.where((d == dec) & (l == label)))) / np.size(np.where(label == l))

print(conf_matrix)

# Plotting Data Distribution Graph
fig = plt.figure()
ax = plt.axes(projection = "3d")
for i in range(total_labels):
    ax.scatter(data[(label==i),1],data[(label==i),2],data[(label==i),3], label=i)
plt.xlabel('X3')
plt.ylabel('X1')
ax.set_zlabel('X2')
ax.legend()
plt.title('Data Distribution Graph')
plt.show()
