"""
Ex. No: 1 - SOLVING XOR PROBLEM USING DNN
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset & Architecture
X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y = np.array([[1], [1], [0], [0]])

num_i_units, num_h_units, num_o_units = 2, 2, 1
learning_rate = 0.5
reg_param = 0.0
max_iter = 5000
m = len(X)

# 2. Reproducibility & Initialization
np.random.seed(1)
W1 = np.random.normal(0, 1, (num_h_units, num_i_units))
W2 = np.random.normal(0, 1, (num_o_units, num_h_units))
B1 = np.random.random((num_h_units, 1))
B2 = np.random.random((num_o_units, 1))

def sigmoid(z, derv=False):
    if derv:
        return z * (1 - z)
    return 1 / (1 + np.exp(-z))

def forward(x, W1, W2, B1, B2):
    a0 = x.reshape(x.shape[0], 1)
    z1 = W1.dot(a0) + B1
    a1 = sigmoid(z1)
    z2 = W2.dot(a1) + B2
    a2 = sigmoid(z2)
    return a0, a1, a2

# 3. Training Loop
cost = np.zeros((max_iter, 1))
for i in range(max_iter):
    c = 0.0
    dW1 = np.zeros_like(W1)
    dW2 = np.zeros_like(W2)
    dB1 = np.zeros_like(B1)
    dB2 = np.zeros_like(B2)
    
    for j in range(m):
        a0, a1, a2 = forward(X[j], W1, W2, B1, B2)
        dz2 = a2 - y[j]
        dW2 += dz2.dot(a1.T)
        dB2 += dz2
        dz1 = np.multiply(W2.T.dot(dz2), sigmoid(a1, derv=True))
        dW1 += dz1.dot(a0.T)
        dB1 += dz1
        
        eps = 1e-15
        a2_clipped = np.clip(a2, eps, 1 - eps)
        c += (-(y[j] * np.log(a2_clipped) + (1 - y[j]) * np.log(1 - a2_clipped))).item()
        
    W1 -= learning_rate * ((dW1 / m) + (reg_param / m) * W1)
    W2 -= learning_rate * ((dW2 / m) + (reg_param / m) * W2)
    B1 -= learning_rate * (dB1 / m)
    B2 -= learning_rate * (dB2 / m)
    
    cost[i] = (c / m) + (reg_param / (2 * m)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    if (i + 1) % 500 == 0 or i == max_iter - 1:
        sys.stdout.write(f"\rIteration: {i + 1}/{max_iter} | Loss: {cost[i, 0]:.4f}")
        sys.stdout.flush()

print("\n\nFinal Predictions:")
for j in range(m):
    _, _, pred = forward(X[j], W1, W2, B1, B2)
    print(f"Input: {X[j]} | Target: {y[j][0]} | Output: {pred[0, 0]:.4f} -> Predicted: {int(pred[0, 0] > 0.5)}")

# 4. Plot Loss
plt.figure(figsize=(7, 4))
plt.plot(range(max_iter), cost, color="tab:blue", lw=2)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("XOR Training Loss Curve")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

