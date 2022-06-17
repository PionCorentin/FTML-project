import numpy as np
import matplotlib.pyplot as plt


def generateY(X, theta_star, sigma):
    n = X.shape[0]
    d = X.shape[1]
    epsilon = r.normal(0, sigma, size=(n, 1))
    y = X @ theta_star + epsilon
    return y



if __name__ == "__main__":
    r = np.random.RandomState(10)

    d = 10
    sigma = 0.5
    n = 10000
    theta = r.rand(d).reshape(d, 1)

    X = r.rand(n, d)
    y = generateY(X, theta, sigma)

    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    sigma_estimator = np.linalg.norm(y - X @ theta_hat) ** 2 / (n - d)
    print("Sigma value is {}",format(float(sigma**2)))
    print("Sigma estimator is {}",format(float(sigma_estimator)))



