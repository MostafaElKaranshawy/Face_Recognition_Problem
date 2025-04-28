import numpy as np

class PCA_GHALBAN:

    def __init__(self, alpha):
        self.alpha = alpha
        self.eigenvalues = None
        self.eigenvectors = None
        self.components = None
        self.mean = None
        self.k = None
        self.variance_explained = None


    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_normalized = X - self.mean
        # np.cov needs samples to be cols, so i transposed the matrix
        # covariance_matrix = np.cov(X_normalized, rowvar=False) 


        # self.eigenvectors, self.eigenvalues = np.linalg.eigh(covariance_matrix)


        U, S, Vt = np.linalg.svd(X_normalized, full_matrices=False)
        self.eigenvalues = S**2 / (X.shape[0] - 1)
        self.eigenvectors = Vt.T  # Eigenvectors are the transpose of Vt

        indices = np.argsort(self.eigenvalues)[::-1]  # descending order
        self.eigenvalues = self.eigenvalues[indices]
        self.eigenvectors = self.eigenvectors[:, indices]

        self.calculate_components_needed()
        self.components = self.eigenvectors[:, :self.k]


    def calculate_components_needed(self):
        total_variance = np.sum(self.eigenvalues)
        variance_explained = 0
        self.k = 0

        for i in range(len(self.eigenvalues)):
            variance_explained += self.eigenvalues[i]
            if variance_explained / total_variance >= self.alpha:
                self.variance_explained = variance_explained
                self.k = i + 1
                break

    # Z = X_centered * W (top eigenvectors)
    def transform(self, X):
        X_normalized = X - self.mean
        return np.dot(X_normalized, self.components) 
    
    # X_reconstructed = Z * W^T + mean
    def inverse_transform(self, Z):
        X_projected = np.dot(Z, self.components.T)
        return X_projected + self.mean