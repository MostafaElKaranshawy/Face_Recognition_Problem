import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
# from kmeans import KMeans
from sklearn.cluster import KMeans


class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4, random_state=42):
        self.K = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.pi = None   # π_k, shape (K,)
        self.means = None     # μ_k, shape (K, D)
        self.covariances = None  # Σ_k, shape (K, D, D)
    
    def _initialize_parameters(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape  # N samples, D features
        
        # Priors: pi[k] = pr(z = k)
        self.pi = np.ones(self.K) / self.K

        # initialize means using KMeans
        kmeans = KMeans(n_clusters=self.K, random_state=self.random_state, n_init=10)
        # kmeans = KMeans(n_clusters=self.K, random_state=self.random_state)
        kmeans.fit(X)  
        self.means = kmeans.cluster_centers_.copy()
        # self.means = kmeans.centroids.copy()

        
        # initialize all covariances to the empirical covariance + small regularization in the diagonal element to avoid singularity
        # rowvar=False is to ensure that we treat each row as a sample and each column as a feature
        emp_cov = np.cov(X, rowvar=False) + 1e-3 * np.eye(D)
        self.covariances = np.array([emp_cov.copy() for _ in range(self.K)])
    
    def _e_step(self, X):
        N, D = X.shape
        log_r = np.zeros((N, self.K))
        
        # Compute log of unnormalized posteriors: log(pi_k) + log(N(x | mu_k, Sigma_k))
        for k in range(self.K):
            log_pi_k = np.log(self.pi[k] + 1e-16)  # log of mixing coefficient
            # Use logpdf to avoid underflow
            log_pdf = multivariate_normal.logpdf(
                X, mean=self.means[k], cov=self.covariances[k], allow_singular=True
            )
            log_r[:, k] = log_pi_k + log_pdf  # Shape (N,)
        
        # Compute log of the denominator: log(sum_k exp(log_r[i, k]))
        log_r_sum = logsumexp(log_r, axis=1, keepdims=True)  # Shape (N, 1)
        
        # r[i, k] = exp(log_r[i, k] - log_r_sum[i])
        r = np.exp(log_r - log_r_sum)  # Shape (N, K)
        
        return r

    def _m_step(self, X, r):
        # r is the responsibilities matrix, shape (N, K)

        N, D = X.shape
        K = r.shape[1]
        
        # N_k = ∑_i r[i,k]
        N_k = r.sum(axis=0)    # shape (K,)
        
        #  Mixing weights π_k = N_k / N
        pi_new = N_k / N       # shape (K,)
        
        # Means μ_k = (Σ_i r[i,k] * x[i]) / N_k
        mu_new = np.zeros((K, D))
        for k in range(K):
            # numerator: Σ_i r[i,k] * x[i]
            weighted_sum = np.dot(r[:, k], X)  # shape (D,)
            mu_new[k] = weighted_sum / N_k[k]
        
        # Covariances Σ_k = (1/N_k) Σ_i r[i,k] (x[i]−μ_k)(x[i]−μ_k)^T  + reg
        covs = np.zeros((K, D, D))
        for k in range(K):
            X_centered = X - mu_new[k]               # shape (N, D)
            weighted_X = r[:, k][:, None] * X_centered # shape (N, D)
            cov_k = np.dot(weighted_X.T, X_centered)   # shape (D, D)
            cov_k /= N_k[k]
            cov_k += 1e-3 * np.eye(D)
            covs[k] = cov_k
        
        # write back to the model
        self.pi = pi_new
        self.means = mu_new
        self.covariances = covs
        
    def _compute_log_likelihood(self, X):

        N, D = X.shape
        log_likelihood = 0.0
        for n in range(N):
            temp = 0.0
            for k in range(self.K):
                temp += self.pi[k] * multivariate_normal.pdf(
                    X[n], mean=self.means[k], cov=self.covariances[k], allow_singular=True
                )
            log_likelihood += np.log(temp + 1e-15)
        return log_likelihood
    
    def fit(self, X):
        
        X = np.asarray(X)
        self._initialize_parameters(X)

        prev_log_likelihood = None        
        for i in range(self.max_iter):
            # E-step
            r = self._e_step(X)
            
            # M-step
            self._m_step(X, r)
            
            # compute log-likelihood
            current_log_likelihood = self._compute_log_likelihood(X)
            
            # check for convergence
            if prev_log_likelihood is not None and abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                break
            
            prev_log_likelihood = current_log_likelihood
        
        return self
    
    def perform_hard_prediction(self, X):
        r = self._e_step(np.asarray(X))
        return np.argmax(r, axis=1)
    
    def predict_probability(self, X):
        r = self._e_step(np.asarray(X))
        return r