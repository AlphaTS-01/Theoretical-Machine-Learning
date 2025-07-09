import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class LogisticRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.loss_history = []
        self.theta_history = []
        
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.random.normal(0, 0.01, X_b.shape[1])
        
        for i in range(self.n_iter):
            z = X_b.dot(self.theta)
            predictions = self._sigmoid(z)
            
            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            self.loss_history.append(loss)
            self.theta_history.append(self.theta.copy())
            
            errors = predictions - y
            gradients = X_b.T.dot(errors) / len(y)
            
            self.theta -= self.lr * gradients
            
    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(X_b.dot(self.theta))
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

iris = load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = LogisticRegressionGD(lr=0.1, n_iter=1000)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_train_proba = model.predict_proba(X_train)
y_test_proba = model.predict_proba(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Logistic Regression with Gradient Descent - Enhanced Analysis', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict_proba(mesh_points)
Z = Z.reshape(xx.shape)

contour = ax1.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='RdYlBu')
ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)

colors = ['red', 'blue']
labels = ['Setosa', 'Versicolor']
for i in range(2):
    mask = y == i
    ax1.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
               c=colors[i], label=labels[i], s=50, alpha=0.8, edgecolors='black')

ax1.set_xlabel('Sepal Length (Standardized)', fontsize=12)
ax1.set_ylabel('Sepal Width (Standardized)', fontsize=12)
ax1.set_title('Decision Boundary with Probability Contours', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

cbar = plt.colorbar(contour, ax=ax1, shrink=0.8)
cbar.set_label('Probability of Versicolor', fontsize=10)

textstr = f'Train Accuracy: {train_acc:.3f}\nTest Accuracy: {test_acc:.3f}'
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax2 = axes[0, 1]
ax2.plot(model.loss_history, color='purple', linewidth=2)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax2.set_title('Loss Function Over Iterations', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

final_loss = model.loss_history[-1]
ax2.annotate(f'Final Loss: {final_loss:.4f}', 
             xy=(len(model.loss_history)-1, final_loss), 
             xytext=(len(model.loss_history)*0.7, final_loss*1.2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

ax3 = axes[1, 0]

all_proba = np.concatenate([y_train_proba, y_test_proba])
all_labels = np.concatenate([y_train, y_test])

ax3.hist(all_proba[all_labels == 0], bins=20, alpha=0.7, color='red', 
         label='Setosa', density=True, edgecolor='black')
ax3.hist(all_proba[all_labels == 1], bins=20, alpha=0.7, color='blue', 
         label='Versicolor', density=True, edgecolor='black')
ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')

ax3.set_xlabel('Predicted Probability', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
theta_history = np.array(model.theta_history)
ax4.plot(theta_history[:, 0], label='Intercept (θ₀)', linewidth=2)
ax4.plot(theta_history[:, 1], label='Sepal Length (θ₁)', linewidth=2)
ax4.plot(theta_history[:, 2], label='Sepal Width (θ₂)', linewidth=2)

ax4.set_xlabel('Iterations', fontsize=12)
ax4.set_ylabel('Parameter Value', fontsize=12)
ax4.set_title('Parameter Evolution During Training', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

final_params = theta_history[-1]
param_text = f'Final Parameters:\nθ₀: {final_params[0]:.3f}\nθ₁: {final_params[1]:.3f}\nθ₂: {final_params[2]:.3f}'
ax4.text(0.05, 0.95, param_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.show()

print("="*50)
print("LOGISTIC REGRESSION RESULTS")
print("="*50)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Final Loss: {final_loss:.4f}")
print(f"Final Parameters: θ₀={final_params[0]:.4f}, θ₁={final_params[1]:.4f}, θ₂={final_params[2]:.4f}")
print(f"Learning Rate: {model.lr}")
print(f"Iterations: {model.n_iter}")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=labels))