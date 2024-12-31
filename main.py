import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from alpha_vantage.timeseries import TimeSeries
import time

# Generate synthetic data
# y = 3x + 2 + noise
#np.random.seed(42)
x = np.random.rand(100, 1).astype(np.float32)  # 100 data points
y = 3 * x + 2 + np.random.randn(100, 1).astype(np.float32) * 0.1


# Initialize TimeSeries with your API key
ts = TimeSeries(key='38ZWG370RS8AKVJ6', output_format='pandas')

# Function to fetch data with error handling
def fetch_data(symbol):
    try:
        data, meta_data = ts.get_intraday(symbol=symbol, interval='5min', outputsize='full')
        return data
    except Exception as e:  # Catch all exceptions
        print(f"Error fetching data: {e}")
        return None

# Fetch data for IBM
data = fetch_data('IBM')
if data is not None:
    print(data.head())
else:
    print("Failed to retrieve data.")

#data is the array


# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input, one output

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Training loop
epochs = 500
for epoch in range(epochs):
    # Forward pass
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Display the learned parameters
[w, b] = model.parameters()
print(f"Learned weight: {w.item():.2f}, Learned bias: {b.item():.2f}")

# Plot the results
predicted = model(x_tensor).detach().numpy()
plt.scatter(x, y, label='Original Data', color='blue')
plt.plot(x, predicted, label='Fitted Line', color='red')
plt.legend()
plt.show()