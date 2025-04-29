import numpy as np

class KMeans:
    """
    KMeans clustering algorithm implementation.
    This class provides methods to fit the model to the data, predict clusters,
    and evaluate the clustering performance.
    steps:
        - Initialize the centroids.
        - Assign each sample to the nearest centroid.
        - Update the centroids based on the assigned clusters.
        - Repeat the assignment and update steps until convergence.
    """
    def __init__(self, n_clusters, random_state=42):
        self.centroids = []
        self.labels = []
        self.n_clusters = n_clusters
        np.random.seed(random_state)
    
    def fit(self, data):
        prev_centroids = np.array([])
        self.centroids = self._initialize_centroids(data)

        while not np.array_equal(prev_centroids, self.centroids):
            cluster_indices = self._assign_clusters(data, self.centroids)
            prev_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(data, cluster_indices)
            
    def predict(self, data):
        cluster_indices = self._assign_clusters(data, self.centroids)
        return cluster_indices
    
    def _initialize_centroids(self, data):

        indices = np.random.choice(len(data), size=self.n_clusters, replace=False)
        centroids = data[indices]
        return centroids
    
    def _assign_clusters(self, data, centroids):
        cluster_indices = []
        for i in range(len(data)):
            min_distance = np.inf
            cluster_index = -1
            for j in range(len(centroids)):
                distance = self._calculate_distance(data[i], centroids[j])
                if distance < min_distance:
                    min_distance = distance
                    cluster_index = j
            cluster_indices.append(cluster_index)
        return np.array(cluster_indices)

    
    # def _update_centroids(self, data, labels, n_clusters):
    #     new_centroids = []
    #     for i in range(n_clusters):
    #         cluster_data = data[labels == i]
    #         if len(cluster_data) > 0:
    #             new_centroid = np.mean(cluster_data, axis=0)
    #             new_centroids.append(new_centroid)
    #         else:
    #             new_centroids.append(np.zeros(data.shape[1]))
    #     return np.array(new_centroids)
    
    def _update_centroids(self, data, labels):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_data = data[labels == i]
            if len(cluster_data) > 0:
                new_centroid = np.mean(cluster_data, axis=0)
            else:
                # Reinitialize to a random data point if cluster is empty
                random_index = np.random.choice(len(data))
                new_centroid = data[random_index]
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    
    def _calculate_distance(self, a, b):
        distance = np.linalg.norm(a - b)
        return distance
    