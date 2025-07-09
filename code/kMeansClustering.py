import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans as SklearnKMeans
import seaborn as sns

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4, init='random'):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        self.centroid_history = []
        
    def _initialize_centroids(self, X):
        if self.init == 'random':
            # Random initialization
            n_samples, n_features = X.shape
            centroids = np.random.rand(self.k, n_features)
            centroids = centroids * (X.max(axis=0) - X.min(axis=0)) + X.min(axis=0)
        elif self.init == 'k-means++':
            # K-means++ initialization
            centroids = []
            centroids.append(X[np.random.randint(0, X.shape[0])])
            
            for _ in range(1, self.k):
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
                probabilities = distances / distances.sum()
                cumulative_prob = probabilities.cumsum()
                r = np.random.rand()
                
                for j, p in enumerate(cumulative_prob):
                    if r < p:
                        centroids.append(X[j])
                        break
            centroids = np.array(centroids)
        else:
            raise ValueError("init must be 'random' or 'k-means++'")
            
        return centroids
    
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        self.centroid_history = [self.centroids.copy()]
        for iteration in range(self.max_iters):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            new_centroids = np.array([X[self.labels == i].mean(axis=0) if len(X[self.labels == i]) > 0 
                                    else self.centroids[i] for i in range(self.k)])
            inertia = sum([np.sum((X[self.labels == i] - new_centroids[i])**2) 
                          for i in range(self.k)])
            self.inertia_history.append(inertia)
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                print(f"Converged after {iteration + 1} iterations")
                break
            self.centroids = new_centroids
            self.centroid_history.append(self.centroids.copy())
        return self
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels

np.random.seed(42)
X_blobs, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                            random_state=42, cluster_std=1.2)
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
scaler_blobs = StandardScaler()
X_blobs_scaled = scaler_blobs.fit_transform(X_blobs)
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)
X_iris_2d = X_iris_scaled[:, :2]
kmeans_blobs = KMeans(k=4, init='k-means++', max_iters=100)
labels_blobs = kmeans_blobs.fit_predict(X_blobs_scaled)
kmeans_iris = KMeans(k=3, init='k-means++', max_iters=100)
labels_iris = kmeans_iris.fit_predict(X_iris_2d)
k_range = range(1, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    if k == 1:
        inertias.append(np.sum((X_blobs_scaled - X_blobs_scaled.mean(axis=0))**2))
        silhouette_scores.append(0)
    else:
        kmeans_temp = KMeans(k=k, init='k-means++', max_iters=100)
        labels_temp = kmeans_temp.fit_predict(X_blobs_scaled)
        inertias.append(kmeans_temp.inertia_history[-1])
        silhouette_scores.append(silhouette_score(X_blobs_scaled, labels_temp))

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('K-Means Clustering - Enhanced Analysis', fontsize=16, fontweight='bold')
ax1 = axes[0, 0]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
for i in range(4):
    mask = y_true == i
    ax1.scatter(X_blobs_scaled[mask, 0], X_blobs_scaled[mask, 1], 
               c=colors[i], alpha=0.7, s=50, label=f'True Cluster {i}')
ax1.set_xlabel('Feature 1', fontsize=12)
ax1.set_ylabel('Feature 2', fontsize=12)
ax1.set_title('Original Blob Data (True Clusters)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
for i in range(4):
    mask = labels_blobs == i
    ax2.scatter(X_blobs_scaled[mask, 0], X_blobs_scaled[mask, 1], 
               c=colors[i], alpha=0.7, s=50, label=f'Cluster {i}')
ax2.scatter(kmeans_blobs.centroids[:, 0], kmeans_blobs.centroids[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='Centroids')
ax2.set_xlabel('Feature 1', fontsize=12)
ax2.set_ylabel('Feature 2', fontsize=12)
ax2.set_title('K-Means Clustering Result (k=4)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

silhouette_blob = silhouette_score(X_blobs_scaled, labels_blobs)
ax2.text(0.02, 0.98, f'Silhouette Score: {silhouette_blob:.3f}', 
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax3 = axes[0, 2]
ax3.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
ax3.set_ylabel('Inertia (WCSS)', fontsize=12)
ax3.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

ax3.annotate('Elbow Point', xy=(4, inertias[3]), xytext=(6, inertias[3] + 50),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold')

ax4 = axes[1, 0]
ax4.plot(k_range[1:], silhouette_scores[1:], 'go-', linewidth=2, markersize=8)
ax4.set_xlabel('Number of Clusters (k)', fontsize=12)
ax4.set_ylabel('Silhouette Score', fontsize=12)
ax4.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

optimal_k = k_range[1:][np.argmax(silhouette_scores[1:])]
ax4.annotate(f'Optimal k={optimal_k}', xy=(optimal_k, max(silhouette_scores[1:])), 
            xytext=(optimal_k + 1, max(silhouette_scores[1:]) - 0.1),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold')

ax5 = axes[1, 1]
ax5.plot(kmeans_blobs.inertia_history, 'mo-', linewidth=2, markersize=6)
ax5.set_xlabel('Iterations', fontsize=12)
ax5.set_ylabel('Inertia', fontsize=12)
ax5.set_title('Convergence History', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

final_inertia = kmeans_blobs.inertia_history[-1]
ax5.annotate(f'Final: {final_inertia:.2f}', 
            xy=(len(kmeans_blobs.inertia_history)-1, final_inertia),
            xytext=(len(kmeans_blobs.inertia_history)*0.7, final_inertia*1.1),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

ax6 = axes[1, 2]
for i in range(4):
    mask = labels_blobs == i
    ax6.scatter(X_blobs_scaled[mask, 0], X_blobs_scaled[mask, 1], 
               c=colors[i], alpha=0.3, s=30)

for i in range(min(5, len(kmeans_blobs.centroid_history))):
    centroids = kmeans_blobs.centroid_history[i]
    ax6.scatter(centroids[:, 0], centroids[:, 1], 
               c='black', marker='x', s=100, alpha=0.7)
    if i > 0:
        for j in range(4):
            ax6.plot([kmeans_blobs.centroid_history[i-1][j, 0], centroids[j, 0]],
                    [kmeans_blobs.centroid_history[i-1][j, 1], centroids[j, 1]],
                    'k--', alpha=0.5)

ax6.set_xlabel('Feature 1', fontsize=12)
ax6.set_ylabel('Feature 2', fontsize=12)
ax6.set_title('Centroid Movement During Training', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

ax7 = axes[2, 0]
target_names = iris.target_names
for i, target_name in enumerate(target_names):
    mask = y_iris == i
    ax7.scatter(X_iris_2d[mask, 0], X_iris_2d[mask, 1], 
               c=colors[i], alpha=0.7, s=50, label=target_name)
ax7.set_xlabel('Sepal Length (Standardized)', fontsize=12)
ax7.set_ylabel('Sepal Width (Standardized)', fontsize=12)
ax7.set_title('Iris Dataset (True Labels)', fontsize=14, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

ax8 = axes[2, 1]
for i in range(3):
    mask = labels_iris == i
    ax8.scatter(X_iris_2d[mask, 0], X_iris_2d[mask, 1], 
               c=colors[i], alpha=0.7, s=50, label=f'Cluster {i}')
ax8.scatter(kmeans_iris.centroids[:, 0], kmeans_iris.centroids[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='Centroids')
ax8.set_xlabel('Sepal Length (Standardized)', fontsize=12)
ax8.set_ylabel('Sepal Width (Standardized)', fontsize=12)
ax8.set_title('Iris Dataset - K-Means Result (k=3)', fontsize=14, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

ari_iris = adjusted_rand_score(y_iris, labels_iris)
silhouette_iris = silhouette_score(X_iris_2d, labels_iris)
ax8.text(0.02, 0.98, f'ARI: {ari_iris:.3f}\nSilhouette: {silhouette_iris:.3f}',
         transform=ax8.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))