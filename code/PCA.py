import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA
import seaborn as sns

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigenvalues)
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered.dot(self.components)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed):
        return X_transformed.dot(self.components.T) + self.mean

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_custom = PCA(n_components=2)
X_pca_custom = pca_custom.fit_transform(X_scaled)

pca_sklearn = SklearnPCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)

pca_full = PCA(n_components=4)
pca_full.fit(X_scaled)

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Principal Component Analysis (PCA) - Enhanced Analysis', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
colors = ['red', 'blue', 'green']
for i, target_name in enumerate(target_names):
    mask = y == i
    ax1.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
               c=colors[i], label=target_name, alpha=0.7, s=50)
ax1.set_xlabel('Sepal Length (Standardized)', fontsize=12)
ax1.set_ylabel('Sepal Width (Standardized)', fontsize=12)
ax1.set_title('Original Data (First 2 Features)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
for i, target_name in enumerate(target_names):
    mask = y == i
    ax2.scatter(X_pca_custom[mask, 0], X_pca_custom[mask, 1], 
               c=colors[i], label=target_name, alpha=0.7, s=50)
ax2.set_xlabel(f'PC1 ({pca_custom.explained_variance_ratio[0]:.1%} variance)', fontsize=12)
ax2.set_ylabel(f'PC2 ({pca_custom.explained_variance_ratio[1]:.1%} variance)', fontsize=12)
ax2.set_title('PCA Transformed Data', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[0, 2]
components = ['PC1', 'PC2', 'PC3', 'PC4']
ax3.bar(components, pca_full.explained_variance_ratio, 
        color=['darkblue', 'blue', 'lightblue', 'lightcyan'], alpha=0.8)
ax3.set_ylabel('Explained Variance Ratio', fontsize=12)
ax3.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(pca_full.explained_variance_ratio):
    ax3.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')

cumulative_var = np.cumsum(pca_full.explained_variance_ratio)
ax3_twin = ax3.twinx()
ax3_twin.plot(components, cumulative_var, color='red', marker='o', linewidth=2, markersize=8)
ax3_twin.set_ylabel('Cumulative Explained Variance', fontsize=12, color='red')
ax3_twin.tick_params(axis='y', labelcolor='red')

ax4 = axes[1, 0]
loadings = pca_custom.components.T
im = ax4.imshow(loadings, cmap='RdBu', aspect='auto')
ax4.set_xticks(range(len(components[:2])))
ax4.set_xticklabels(['PC1', 'PC2'])
ax4.set_yticks(range(len(feature_names)))
ax4.set_yticklabels(feature_names)
ax4.set_title('Principal Components Loadings', fontsize=14, fontweight='bold')

for i in range(len(feature_names)):
    for j in range(2):
        ax4.text(j, i, f'{loadings[i, j]:.2f}', ha='center', va='center', 
                color='white' if abs(loadings[i, j]) > 0.5 else 'black', fontweight='bold')

plt.colorbar(im, ax=ax4, shrink=0.8)

ax5 = axes[1, 1]
n_components_range = range(1, 5)
reconstruction_errors = []

for n in n_components_range:
    pca_temp = PCA(n_components=n)
    X_transformed = pca_temp.fit_transform(X_scaled)
    X_reconstructed = pca_temp.inverse_transform(X_transformed)
    error = np.mean((X_scaled - X_reconstructed) ** 2)
    reconstruction_errors.append(error)

ax5.plot(n_components_range, reconstruction_errors, 'o-', linewidth=2, markersize=8, color='purple')
ax5.set_xlabel('Number of Components', fontsize=12)
ax5.set_ylabel('Mean Squared Reconstruction Error', fontsize=12)
ax5.set_title('Reconstruction Error vs Components', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

for i, error in enumerate(reconstruction_errors):
    ax5.annotate(f'{error:.4f}', (i+1, error), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=10)

ax6 = axes[1, 2]
for i, target_name in enumerate(target_names):
    mask = y == i
    ax6.scatter(X_pca_custom[mask, 0], X_pca_custom[mask, 1], 
               c=colors[i], label=target_name, alpha=0.7, s=50)

feature_vectors = pca_custom.components.T
scale_factor = 3 
for i, feature in enumerate(feature_names):
    ax6.arrow(0, 0, feature_vectors[i, 0] * scale_factor, 
              feature_vectors[i, 1] * scale_factor,
              head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.8)
    ax6.text(feature_vectors[i, 0] * scale_factor * 1.1, 
             feature_vectors[i, 1] * scale_factor * 1.1,
             feature.replace(' (cm)', ''), fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax6.set_xlabel(f'PC1 ({pca_custom.explained_variance_ratio[0]:.1%} variance)', fontsize=12)
ax6.set_ylabel(f'PC2 ({pca_custom.explained_variance_ratio[1]:.1%} variance)', fontsize=12)
ax6.set_title('PCA Biplot (Data + Feature Vectors)', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = np.corrcoef(X_scaled, rowvar=False)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=feature_names, yticklabels=feature_names)
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("="*60)
print("PRINCIPAL COMPONENT ANALYSIS RESULTS")
print("="*60)
print(f"Original data shape: {X.shape}")
print(f"Transformed data shape: {X_pca_custom.shape}")
print(f"Total variance explained by PC1 and PC2: {sum(pca_custom.explained_variance_ratio):.1%}")
print(f"Individual explained variance ratios: {pca_custom.explained_variance_ratio}")
print(f"Eigenvalues: {pca_custom.explained_variance}")
print("\nPrincipal Components (Loadings):")
for i, component in enumerate(pca_custom.components):
    print(f"PC{i+1}: {component}")
print(f"\nFeature importance in PC1: {dict(zip(feature_names, abs(pca_custom.components[0])))}")
print(f"Feature importance in PC2: {dict(zip(feature_names, abs(pca_custom.components[1])))}")
print(f"\nReconstruction error with 2 components: {reconstruction_errors[1]:.6f}")
print(f"Data reduction: {X.shape[1]} → {pca_custom.n_components} dimensions")