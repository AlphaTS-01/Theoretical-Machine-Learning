import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2), axis=1)
    
    def _calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        else:
            raise ValueError("Unsupported distance metric")
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._calculate_distance(x, self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            prediction = np.bincount(k_nearest_labels).argmax()
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        probabilities = []
        for x in X:
            distances = self._calculate_distance(x, self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            unique_labels = np.unique(self.y_train)
            proba = np.zeros(len(unique_labels))
            for i, label in enumerate(unique_labels):
                proba[i] = np.sum(k_nearest_labels == label) / self.k
            probabilities.append(proba)
            
        return np.array(probabilities)

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_2d = X[:, :2]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_2d_scaled = scaler.fit_transform(X_2d)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d_scaled, y, test_size=0.3, random_state=42)

k_values = [1, 3, 5, 7, 9, 11, 15, 19]
accuracies = []
models = {}

for k in k_values:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    models[k] = knn

optimal_k = k_values[np.argmax(accuracies)]
optimal_model = models[optimal_k]

knn_2d = KNN(k=optimal_k)
knn_2d.fit(X_train_2d, y_train_2d)

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('K-Nearest Neighbors (KNN) - Enhanced Analysis', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
colors = ['red', 'blue', 'green']
k_demo = 5

h = 0.02
x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
y_min, y_max = X_2d_scaled[:, 1].min() - 1, X_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

knn_viz = KNN(k=k_demo)
knn_viz.fit(X_2d_scaled, y)
Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax1.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
ax1.contour(xx, yy, Z, colors='black', linestyles='--', linewidths=0.5)

for i, target_name in enumerate(target_names):
    mask = y == i
    ax1.scatter(X_2d_scaled[mask, 0], X_2d_scaled[mask, 1], 
               c=colors[i], label=target_name, alpha=0.8, s=50, edgecolors='black')

ax1.set_xlabel('Sepal Length (Standardized)', fontsize=12)
ax1.set_ylabel('Sepal Width (Standardized)', fontsize=12)
ax1.set_title(f'Decision Boundary (k={k_demo})', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8, color='purple')
ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
ax2.set_xlabel('K Value', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Model Performance vs K Value', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

for i, (k, acc) in enumerate(zip(k_values, accuracies)):
    ax2.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9)

ax3 = axes[0, 2]
distance_metrics = ['euclidean', 'manhattan']
metric_accuracies = []

for metric in distance_metrics:
    knn_metric = KNN(k=optimal_k, distance_metric=metric)
    knn_metric.fit(X_train, y_train)
    y_pred_metric = knn_metric.predict(X_test)
    acc_metric = accuracy_score(y_test, y_pred_metric)
    metric_accuracies.append(acc_metric)

bars = ax3.bar(distance_metrics, metric_accuracies, color=['skyblue', 'lightcoral'], alpha=0.8)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('Distance Metrics Comparison', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, metric_accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

ax4 = axes[1, 0]
y_pred_optimal = optimal_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, ax=ax4)
ax4.set_title(f'Confusion Matrix (k={optimal_k})', fontsize=14, fontweight='bold')
ax4.set_ylabel('True Label', fontsize=12)
ax4.set_xlabel('Predicted Label', fontsize=12)

ax5 = axes[1, 1]
sample_idx = 5
sample_point = X_test_2d[sample_idx]
sample_label = y_test_2d[sample_idx]

distances = knn_2d._calculate_distance(sample_point, X_train_2d)
k_indices = np.argsort(distances)[:optimal_k]
k_nearest_points = X_train_2d[k_indices]
k_nearest_labels = y_train_2d[k_indices]

for i, target_name in enumerate(target_names):
    mask = y_train_2d == i
    ax5.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], 
               c=colors[i], alpha=0.3, s=30, label=f'{target_name} (train)')

for i, (point, label) in enumerate(zip(k_nearest_points, k_nearest_labels)):
    ax5.scatter(point[0], point[1], c=colors[label], s=100, 
               edgecolors='black', linewidth=2, marker='s')
    ax5.plot([sample_point[0], point[0]], [sample_point[1], point[1]], 
             'k--', alpha=0.5, linewidth=1)

ax5.scatter(sample_point[0], sample_point[1], c='red', s=200, 
           marker='*', edgecolors='black', linewidth=2, label='Query Point')

ax5.set_xlabel('Sepal Length (Standardized)', fontsize=12)
ax5.set_ylabel('Sepal Width (Standardized)', fontsize=12)
ax5.set_title(f'K-Nearest Neighbors Analysis (k={optimal_k})', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = axes[1, 2]
probabilities = optimal_model.predict_proba(X_test)
prob_df = probabilities

bottom = np.zeros(len(X_test))
for i, target_name in enumerate(target_names):
    ax6.bar(range(len(X_test)), prob_df[:, i], bottom=bottom, 
           label=target_name, color=colors[i], alpha=0.8)
    bottom += prob_df[:, i]

ax6.set_xlabel('Test Sample Index', fontsize=12)
ax6.set_ylabel('Probability', fontsize=12)
ax6.set_title('Class Probability Predictions', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

test_acc = accuracy_score(y_test, y_pred_optimal)
ax6.text(0.02, 0.95, f'Test Accuracy: {test_acc:.3f}', transform=ax6.transAxes, 
         fontsize=11, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

sklearn_knn = KNeighborsClassifier(n_neighbors=optimal_k)
sklearn_knn.fit(X_train, y_train)
sklearn_pred = sklearn_knn.predict(X_test)
sklearn_acc = accuracy_score(y_test, sklearn_pred)

print("="*60)
print("K-NEAREST NEIGHBORS RESULTS")
print("="*60)
print(f"Optimal k value: {optimal_k}")
print(f"Custom KNN accuracy: {test_acc:.4f}")
print(f"Sklearn KNN accuracy: {sklearn_acc:.4f}")
print(f"Distance metric: {optimal_model.distance_metric}")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")
print("\nAccuracy for different k values:")
for k, acc in zip(k_values, accuracies):
    print(f"k={k}: {acc:.4f}")
print(f"\nDistance metrics comparison:")
for metric, acc in zip(distance_metrics, metric_accuracies):
    print(f"{metric}: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal, target_names=target_names))