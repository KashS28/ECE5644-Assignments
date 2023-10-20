from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy import linalg as LA

np.set_printoptions(threshold=np.inf)
plt.rcParams['figure.figsize'] = [9,9]

# Total features
feature = 4                  

# Total Samples
sample = 10000          

# Total classes
label = 2                   

# Mean vectors
matrix_mean = np.ones(shape=[label, feature])
matrix_mean [0, :] = [-1,-1,-1,-1]

# Covariance matrices
covariance_matrix = np.ones(shape=[label, feature, feature])
covariance_matrix [0, :, :] = [[2, -0.5, 0.3, 0],[-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]]
covariance_matrix [1, :, :] = [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]

np.random.seed(47) 

# Class prior and assigning labels
classprior = [0.7, 0.3]
label = (np.random.rand(sample) >= classprior[1]).astype(int)

# Generating gaussian distribution
X = np.zeros(shape = [sample, feature])
for i in range(sample): 
        if (label[i] == 0):
                X[i, :] = np.random.multivariate_normal(matrix_mean[0, :], covariance_matrix[0, :,:])
        elif (label[i] == 1):
                X[i, :] = np.random.multivariate_normal(matrix_mean[1, :], covariance_matrix[1, :,:])

# Computing between-class and within-class scatter matrices
Sb = (matrix_mean[0, :] - matrix_mean[1, :]) * np.transpose (matrix_mean[0, :] - matrix_mean[1, :])
Sw = covariance_matrix[0, :, :] + covariance_matrix[1, :, :]

# Eigenvalues and eigen vectors of inverse(Sw.Sb)
V, W = LA.eig(LA.inv(Sw) * Sb)

# Eigen vector with maximizing optimization objective
W_LDA = W[np.argmax(V)]
X0 = X[np.where(label == 0)]
X1 = X[np.where(label == 1)]

# Data projection using wLDA
Y0 = np.zeros(len(X0))
Y1 = np.zeros(len(X1))
Y0 = np.dot(np.transpose(W_LDA), np.transpose(X0))
Y1 = np.dot(np.transpose(W_LDA), np.transpose(X1))

# Ranging threshold from minimum to maximum
Y = np.concatenate([Y0, Y1])
sort_Y = np.sort(Y)
tausweep = []

# Calculating mid-points
for i in range(0,9999):
        tausweep.append((sort_Y[i] + sort_Y[i+1])/2.0)

# Array initialization
dec = []
tp = [None] * len(tausweep)
fp = [None] * len(tausweep)
minerror = [None] * len(tausweep)

# Classifying for each threshold and compute error and evaluation metrics
for (index, tau) in enumerate(tausweep):
        dec = (Y >= tau)
        tp[index] = (np.size(np.where((dec == 1) & (label == 1))))/np.size(np.where(label == 1))
        fp[index] = (np.size(np.where((dec == 1) & (label == 0))))/np.size(np.where(label == 0))
        minerror[index] = (classprior[0] * fp[index]) + (classprior[1] * (1 - tp[index]))

# Theoretical classification based on class prior
gamma_ideal = classprior[0] / classprior[1]
ideal_dec = (Y >= gamma_ideal)
ideal_tp = (np.size(np.where((ideal_dec == 1) & (label == 1))))/np.size(np.where(label == 1))
ideal_fp = (np.size(np.where((ideal_dec == 1) & (label == 0))))/np.size(np.where(label == 0))
minerror_ideal = (classprior[0] * ideal_fp) + (classprior[1] * (1 - ideal_tp))
print("Tau Ideal - %f Minimum Error %f" %(gamma_ideal, minerror_ideal))

# Plot ROC curve
plt.plot(fp, tp, color = 'blue')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC Curve')
plt.plot(fp[np.argmin(minerror)], tp[np.argmin(minerror)],'o',color = 'brown')
plt.show()

print("Tau Practical - %f Minimum Error %f"%(tausweep[np.argmin(minerror)], np.min(minerror)))
