import random
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)

# Total Samples
samples = 10000 

# Total features
feature = 3         

# Total classes
total_labels = 3 

# Total gaussian distributions
gauss_d = 4          

np.random.seed(47) 

# Class Priors
classprior = np.array([[0.3, 0.3, 0.4]])

# Mean vectors
mean_matrix = np.zeros(shape=[gauss_d, feature])
mean_matrix [0, :] = [0, 0, 15]
mean_matrix [1, :] = [0, 15, 0]
mean_matrix [2, :] = [15, 0, 0]
mean_matrix [3, :] = [15, 0, 15]

# Covariance matrices
covariance_matrix = np.zeros(shape=[gauss_d, feature, feature])
covariance_matrix[0, :, :] = 36 * np.linalg.matrix_power((np.eye(feature)) + (0.01 * np.random.randn(feature, feature)), 2)
covariance_matrix[1, :, :] = 36 * np.linalg.matrix_power((np.eye(feature)) + (0.02 * np.random.randn(feature, feature)), 2)
covariance_matrix[2, :, :] = 36 * np.linalg.matrix_power((np.eye(feature)) + (0.03 * np.random.randn(feature, feature)), 2)
covariance_matrix[3, :, :] = 36 * np.linalg.matrix_power((np.eye(feature)) + (0.04 * np.random.randn(feature, feature)), 2)

prior_gmm_label3 = [0.5,0.5]
cumsum = np.cumsum(classprior)
random_label = np.random.rand(samples)
label = np.zeros(shape = [10000])
for i in range(0,samples-1):
    if random_label[i] <= cumsum[0]:
        label[i] = 0
    elif random_label[i] <= cumsum[1]:
        label[i] = 1
    else:
        label[i] = 2 

# Generating gaussian distribution
X = np.zeros(shape = [samples, feature])
for i in range(samples):
    if (label[i] == 0):
        X[i, :] = np.random.multivariate_normal(mean_matrix[0, :], covariance_matrix[0, :, :])
    elif (label[i] == 1):
        X[i, :] = np.random.multivariate_normal(mean_matrix[1, :], covariance_matrix[1, :, :])
    elif (label[i] == 2):
        if (np.random.rand(1,1) >= prior_gmm_label3[1]):
            X[i, :] = np.random.multivariate_normal(mean_matrix[2, :], covariance_matrix[2, :, :])
        else:
            X[i, :] = np.random.multivariate_normal(mean_matrix[3, :], covariance_matrix[3, :, :])

loss_matrix = np.array([[0, 10, 10], [1, 0, 10], [1, 1, 0]])

print(loss_matrix)

# Computing Class conditional PDF
px_given = np.zeros(shape = [total_labels, samples])
for i in range(total_labels):
    px_given[i, :] = multivariate_normal.pdf(X,mean = mean_matrix[i, :], cov = covariance_matrix[i, :,:])

# Computing Class Posteriors
px = np.matmul(classprior, px_given)
class_post = (px_given * (np.matlib.repmat(np.transpose(classprior), 1, samples))) / np.matlib.repmat(px, total_labels, 1)

# Evaluating Expected risk and decision
exp_risk = np.matmul(loss_matrix, class_post)
dec = np.argmin(exp_risk, axis = 0)
print("Average Expected Risk:", np.sum(np.min(exp_risk, axis = 0)) / samples)

# Estimate Confusion Matrix
conf_matrix = np.zeros(shape = [total_labels, total_labels])

for d in range(total_labels):
    for l in range(total_labels):
        conf_matrix[d, l] = (np.size(np.where((d == dec) & (l == label)))) / np.size(np.where(label==l))

print(conf_matrix)

# Plot Classification results
fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.scatter(X[(label==2) & (dec == 1),0],X[(label==2) & (dec == 1),1],X[(label==2) & (dec == 1),2],color ='red', marker = 'o')
ax.scatter(X[(label==2) & (dec == 2),0],X[(label==2) & (dec == 2),1],X[(label==2) & (dec == 2),2],color ='green', marker = 'o')
ax.scatter(X[(label==2) & (dec == 0),0],X[(label==2) & (dec == 0),1],X[(label==2) & (dec == 0),2],color ='red', marker = 'o')
ax.scatter(X[(label==1) & (dec == 1),0],X[(label==1) & (dec == 1),1],X[(label==1) & (dec == 1),2],color ='green', marker = 's')
ax.scatter(X[(label==1) & (dec == 2),0],X[(label==1) & (dec == 2),1],X[(label==1) & (dec == 2),2],color ='red', marker = 's')
ax.scatter(X[(label==1) & (dec == 0),0],X[(label==1) & (dec == 0),1],X[(label==1) & (dec == 0),2],color ='red', marker = 's')
ax.scatter(X[(label==0) & (dec == 0),0],X[(label==0) & (dec == 0),1],X[(label==0) & (dec == 0),2],color ='green', marker = '^')
ax.scatter(X[(label==0) & (dec == 2),0],X[(label==0) & (dec == 2),1],X[(label==0) & (dec == 2),2],color ='red', marker = '^')
ax.scatter(X[(label==0) & (dec == 1),0],X[(label==0) & (dec == 1),1],X[(label==0) & (dec == 1),2],color ='red', marker = '^')
plt.xlabel('X1')
plt.ylabel('X2')
ax.set_zlabel('X3')
plt.title('Classification Plot:')
plt.show()