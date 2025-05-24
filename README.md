# Face_Recognition_Problem

Apply Unsupervised Algorithms for Face Recognition for AT&T Kaggle Dataset.

## PCA

- Principal components analysis helps reduce the data dimensions from a very high-dimensional space to a lower one.
- Our main parameter is the alpha/explained variance.
- Alpha is essential when computing by adding the eigenvalues of the PCs (eigenvectors). By sorting the PCs in descending order, we always use the eigenvectors that describe a lot about the data first.
- fit(data): helps in computing the eigenvectors and eigenvalues using Singular Value Decomposition (SVD) and determines which eigenvectors to keep according to the alpha parameter. It’s important to note that we do this after decrementing the mean from the data. (to get true variance)
- transform(data): This function projects the given data onto the low-dimensional space computed in the fitting step. We decrement the mean first from the given data, then project onto the space * W (top eigenvectors)
- inverse_transform(data): This function reconstructs the given data, which is of the low-dimensional space, onto the original space by multiplying it with the Transpose of the eigenvectors and adding the mean back to retrieve the data originally.

## Unsupervised Clustering:

###  K-Means Clustering:

#### Algorithm Explanation:
- Initialize the random centroids of the K clusters (random indices from the train points)
- Assign the points to the cluster with the least distance to its centroid
- Distance is the norm for all features.
- Reconstruct the centroids according to the data points assignments
- If there is a cluster that has no data points assigned to it, choose a random point from the data to be its centroid → to be sure that each cluster has at least one data point assigned to it.
- Repeat the previous steps until convergence → the new centroids are the same as the previous (the centroids don’t change)

#### Performance

![image](https://github.com/user-attachments/assets/a41d6759-43d0-40af-a6d2-123106d137e4)

#### Relation between alpha and classification accuracy

![image](https://github.com/user-attachments/assets/b542047d-ac7f-41df-81dc-93a1d791484b)

As alpha increases, the accuracy decreases because in higher dimensions, data points become sparse (very far from each other due to the curse of dimensionality). This makes the computations of the distances to each cluster centroid increase → more competitions between clusters to assign the points.

#### Relation between K and classification accuracy

![image](https://github.com/user-attachments/assets/57f6c6d8-149c-43dc-b8c5-53a6074ac650)

As K increases the accuracy increases too, as many clusters increase the probability that a label appears in these clusters.
- If K < N→ there will be at least (N - K) labels that won't appear in the mapping of these clusters, so the accuracy will decrease.
- if K >= N → the number of missing labels will decrease and may tend to 0, so the accuracy will increase.

### Gaussian Mixture Model Clustering:

#### Algorithm Explanation:

- We initialize the means of the gaussians by the cluster centroids obtained from KMeans and initialize the covariance matrix with the empirical covariance of the data
- After that we perform the expectation step, where we compute the responsibilities given the current parameters (the probabilities of each point to belong to a certain gaussian) using this formula
  
![image](https://github.com/user-attachments/assets/2c2d9df6-d819-464e-835e-0f0c78e0c494)
  
But since we have a large number of features in the data, we had to take the log of this equation while calculating the responsibilities and take power e to the results at the end of the expectation step
- After that we perform the maximization step to re-estimate the parameters given current responsibilities


  ![image](https://github.com/user-attachments/assets/ccbbcc9c-6e9d-4781-a1c2-00dd1780461a)


- We repeat these two steps until convergence or until we reach the maximum number of iterations, convergence is determined by the log likelihood of the current iteration be within a small tolerance from the log likelihood of the previous iteration, computing the log likelihood using this formula:

![image](https://github.com/user-attachments/assets/c90fd214-ad4e-4fd9-b320-7179a38dd33e)

#### Performance:

![image](https://github.com/user-attachments/assets/8855cc2d-2aa5-488f-a2b1-4788f34a486c)

![image](https://github.com/user-attachments/assets/20d978e4-aed1-4a90-b53b-887bc03cb975)

![image](https://github.com/user-attachments/assets/0910207c-1080-4e5f-ae23-7d47b82bc1a3)

- Relation between alpha and classification accuracy: as alpha increases, the accuracy decreases because in higher dimensions, data points become sparse (very far from each other due to the curse of dimensionality). This makes it harder for the GMM to find meaningful clusters

![image](https://github.com/user-attachments/assets/5dabd15f-9f58-4071-9872-637e0254205c)

![image](https://github.com/user-attachments/assets/6157d9ba-7a0f-43cc-be83-9fa75ba3cb65)

- Relation between K and classification accuracy: as K increases, the model becomes more flexible and more accurate since lower K means more samples per cluster and using the majority vote to determine the accuracy leads to a lot of samples to be errors. So increasing K makes it possible for each cluster to be more pure so less samples as errors so the accuracy increases


## Autoencoder:

Autoencoder is a neural network that learns, in an unsupervised way, a better representation of data in a lower-dimensional space than the input dimension.
Autoencoder consists of:
- Encoder
- Decoder

Input → Encoder → Bottleneck → Decoder → Output
1. Encoder (fully connected layers)
  a. compresses the input into a lower-dimensional latent space, aka the bottleneck
  b. We used a ReLU activation function for non-linearity
2. Decoder
  a. Reconstructs the original input from this bottleneck representation
3. We consider the bottleneck → the latent features/the features that describe the most about our input data

Hyperparameters considered:
- Hidden layer dimension
- Bottleneck layer dimension
- Learning rate
- Batch size
- Epoches number
- 
After tuning the hyperparameters, the best picks are:
- Hidden: 1024
- Bottleneck: 128
- Learning rate: 0.001
- Batch size: 128/256
- Epoches number: 500-1000

#### Auto Encoders with Kmeans

![image](https://github.com/user-attachments/assets/eba3b281-1aa8-47f9-aaeb-6b2bdc7fc65b)

#### Auto Encoders with GMM

![image](https://github.com/user-attachments/assets/609a8714-f23b-43e0-aad7-44a5b97b3a10)


[Dataset](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)
