import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(47)

# Given parameters
samples = 10000
features = 4
num_labels = 2
priors = [0.35, 0.65]
matrix_mean = np.ones(shape=(num_labels, features))
matrix_mean[0, :] = [-1, -1, -1, -1]
covariance_matrix = np.ones(shape=(num_labels, features, features))
covariance_matrix[0, :, :] = [[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]]
covariance_matrix[1, :, :] = [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]

# Generating samples
X = np.zeros(shape=(samples, features))
labels = np.random.rand(samples) >= priors[0]
for i in range(samples):
    if labels[i] == 0:
        X[i, :] = np.random.multivariate_normal(matrix_mean[0, :], covariance_matrix[0, :, :])
    elif labels[i] == 1:
        X[i, :] = np.random.multivariate_normal(matrix_mean[1, :], covariance_matrix[1, :, :])

# Calculating discriminant score
pdf = np.log(multivariate_normal.pdf(X, mean=matrix_mean[0, :], cov=covariance_matrix[0, :, :]))
pdf1 = np.log(multivariate_normal.pdf(X, mean=matrix_mean[1, :], cov=covariance_matrix[1, :, :]))
disc_score = pdf1 - pdf 
ds_sort = np.sort(disc_score)
tausweep = [(ds_sort[t] + ds_sort[t + 1]) / 2.0 for t in range(0, samples - 1)]
TP, FP, minerror = [], [], []

for (i, tau) in enumerate(tausweep):
    dec = (disc_score >= tau)
    TP.append((np.size(np.where((dec == 1) & (labels == 1)))) / np.size(np.where(labels == 1)))
    FP.append((np.size(np.where((dec == 1) & (labels == 0)))) / np.size(np.where(labels == 0)))
    minerror.append((priors[0] * FP[i]) + (priors[1] * (1 - TP[i])))

loggamma_ideal = np.log(priors[0] / priors[1])
ideal_dec = (disc_score >= loggamma_ideal)
ideal_tp = (np.size(np.where((ideal_dec == 1) & (labels == 1)))) / np.size(np.where(labels == 1))
ideal_fp = (np.size(np.where((ideal_dec == 1) & (labels == 0)))) / np.size(np.where(labels == 0))
ideal_minerror = (priors[0] * ideal_fp) + (priors[1] * (ideal_tp))
print("γ Ideal - %f \n Minimum Error %f" % (np.exp(loggamma_ideal), ideal_minerror))

# Plotting Graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Class0 = ax.scatter(X[(labels == 0), 3], X[(labels == 0), 1], X[(labels == 0), 2], '+', color='orange', label="0")
Class1 = ax.scatter(X[(labels == 1), 3], X[labels == 1, 1], X[labels == 1, 2], '.', color='green', label="1")
ax.set_xlabel('X3')
ax.set_ylabel('X1')
ax.set_zlabel('X2')
ax.legend()
plt.title('Generated Data')
plt.show()

# ROC curve
plt.plot(FP, TP, color='pink')
plt.ylabel('True Positive')
plt.xlabel('False Positive')
plt.title('ROC Curve [Minimum Expected Risk Classifier]')
plt.plot(FP[np.argmin(minerror)], TP[np.argmin(minerror)], 'o', color='purple')
plt.show()

print("γ Practical - %f \n Minimum Error %f" % (np.exp(tausweep[np.argmin(minerror)]), np.min(minerror)))
