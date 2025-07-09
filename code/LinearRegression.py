import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.cost_history = []
        self.theta_history = []
        
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.random.normal(0, 0.01, X_b.shape[1])
        
        for i in range(self.n_iter):
            predictions = X_b.dot(self.theta)
            errors = predictions - y
            
            cost = np.mean(errors**2) / 2
            self.cost_history.append(cost)
            self.theta_history.append(self.theta.copy())
            
            gradients = X_b.T.dot(errors) / len(y)
            
            self.theta -= self.lr * gradients
            
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

# Load and prepare data
X, y = datasets.load_diabetes(return_X_y=True)
X_feat = X[:, np.newaxis, 2]  # Use BMI feature

# Standardize features
scaler = StandardScaler()
X_feat_scaled = scaler.fit_transform(X_feat)

# Split data for visualization
train_size = int(0.8 * len(X_feat_scaled))
X_train, X_test = X_feat_scaled[:train_size], X_feat_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train model
model = LinearRegressionGD(lr=0.1, n_iter=500)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Create enhanced visualization
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Linear Regression with Gradient Descent - Enhanced Analysis', fontsize=16, fontweight='bold')

# 1. Training and Test Data with Regression Line
ax1 = axes[0, 0]
ax1.scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data', s=30)
ax1.scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data', s=30)

# Create smooth line for regression
X_line = np.linspace(X_feat_scaled.min(), X_feat_scaled.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
ax1.plot(X_line, y_line, color='green', linewidth=2, label='Regression Line')

ax1.set_xlabel('BMI (Standardized)', fontsize=12)
ax1.set_ylabel('Diabetes Progression', fontsize=12)
ax1.set_title('Linear Regression Fit', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add metrics as text
textstr = f'Train R²: {train_r2:.3f}\nTest R²: {test_r2:.3f}\nTrain MSE: {train_mse:.1f}\nTest MSE: {test_mse:.1f}'
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 2. Cost History
ax2 = axes[0, 1]
ax2.plot(model.cost_history, color='purple', linewidth=2)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Cost (MSE/2)', fontsize=12)
ax2.set_title('Cost Function Over Iterations', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add final cost value
final_cost = model.cost_history[-1]
ax2.annotate(f'Final Cost: {final_cost:.2f}', 
             xy=(len(model.cost_history)-1, final_cost), 
             xytext=(len(model.cost_history)*0.7, final_cost*1.2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# 3. Residuals Plot
ax3 = axes[1, 0]
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

ax3.scatter(y_train_pred, residuals_train, alpha=0.6, color='blue', label='Training', s=30)
ax3.scatter(y_test_pred, residuals_test, alpha=0.6, color='red', label='Test', s=30)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.8)
ax3.set_xlabel('Predicted Values', fontsize=12)
ax3.set_ylabel('Residuals', fontsize=12)
ax3.set_title('Residuals Plot', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Parameter Evolution
ax4 = axes[1, 1]
theta_history = np.array(model.theta_history)
ax4.plot(theta_history[:, 0], label='Intercept (θ₀)', linewidth=2)
ax4.plot(theta_history[:, 1], label='Slope (θ₁)', linewidth=2)
ax4.set_xlabel('Iterations', fontsize=12)
ax4.set_ylabel('Parameter Value', fontsize=12)
ax4.set_title('Parameter Evolution During Training', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add final parameter values
final_params = theta_history[-1]
ax4.text(0.05, 0.95, f'Final θ₀: {final_params[0]:.3f}\nFinal θ₁: {final_params[1]:.3f}', 
         transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

# Print final results
print("="*50)
print("LINEAR REGRESSION RESULTS")
print("="*50)
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Final Parameters: θ₀={final_params[0]:.4f}, θ₁={final_params[1]:.4f}")
print(f"Learning Rate: {model.lr}")
print(f"Iterations: {model.n_iter}")