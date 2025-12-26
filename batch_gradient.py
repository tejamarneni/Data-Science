import numpy as np

# 1. Setup Data (Same as before)
X = np.array([[2100, 3], [1600, 3], [2400, 4]]) # Features
y = np.array([[400], [330], [450]])             # Target

# Scale features! (Crucial for Gradient Descent, unlike Normal Eq)
# If we don't scale, the gradients for '2100' will be huge vs '3'
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# 2. Add Bias (Intercept) column of 1s
m = len(X)
X_b = np.c_[np.ones((m, 1)), X_scaled]

# 3. Initialization
# Start Beta with random numbers or zeros
beta = np.random.randn(3, 1)  
learning_rate = 0.1
n_iterations = 1000

# 4. The Loop (The "Descent")
for iteration in range(n_iterations):
    
    # A. Calculate Predictions (Forward pass)
    predictions = X_b.dot(beta)
    
    # B. Calculate Errors
    errors = predictions - y
    
    # C. Calculate Gradient (The Compass)
    # Math: (2/m) * X^T * (X*beta - y)
    gradients = (2/m) * X_b.T.dot(errors)
    
    # D. Update Beta (Take the step)
    # Math: beta_new = beta_old - learning_rate * gradient
    beta = beta - learning_rate * gradients
    
    # Optional: Print cost every 100 steps to verify we are "descending"
    if iteration % 100 == 0:
        cost = np.mean(errors**2)
        print(f"Iteration {iteration}: Cost (MSE) = {cost:.4f}")

print("-" * 30)
print("Final Beta values (scaled):")
print(beta)
