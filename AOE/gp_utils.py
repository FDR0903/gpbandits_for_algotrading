import numpy as np
import scipy

def Wasserstein_GP(mean1, cov1, mean2, cov2):

    mu_0 = mean1
    K_0 = cov1

    mu_1 = mean2
    K_1 = cov2

    sqrtK_0 = scipy.linalg.sqrtm(K_0)
    first_term = np.dot(sqrtK_0, K_1)
    K_0_K_1_K_0 = np.dot(first_term, sqrtK_0)

    cov_dist = np.trace(K_0) + np.trace(K_1) - 2 * np.trace(scipy.linalg.sqrtm(K_0_K_1_K_0))
    l2norm = (np.sum(np.square(abs(mu_0 - mu_1))))
    d = np.real(np.sqrt(l2norm + cov_dist))

    return d

def Wasserstein_GP_mean(mean1, cov1, mean2, cov2):
    n = mean1.shape[0]

    mu_0 = mean1
    K_0 = cov1

    mu_1 = mean2
    K_1 = cov2

    sqrtK_0 = scipy.linalg.sqrtm(K_0)
    first_term = np.dot(sqrtK_0, K_1)
    K_0_K_1_K_0 = np.dot(first_term, sqrtK_0)

    cov_dist = np.trace(K_0) + np.trace(K_1) - 2 * np.trace(scipy.linalg.sqrtm(K_0_K_1_K_0))
    l2norm = (np.mean(np.square(abs(mu_0 - mu_1))))
    d = np.real(l2norm + cov_dist/n)

    return d