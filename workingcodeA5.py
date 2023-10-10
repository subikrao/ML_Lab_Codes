customer_data = {
    'Candies': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'Mangoes': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'Milk_Packets': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'Payment': [1, 1, 1, 0, 1, 0, 1, 1, 0, 0],  # Encoding Yes=1, No=0
    'High_Value': [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]  # Encoding Yes=1, No=0
}

df = pd.DataFrame(customer_data)

df[['Candies', 'Mangoes', 'Milk_Packets']] = (df[['Candies', 'Mangoes', 'Milk_Packets']] - df[['Candies', 'Mangoes', 'Milk_Packets']].mean()) / df[['Candies', 'Mangoes', 'Milk_Packets']].std()


customer_learning_rate = 0.01

# Training Loop
customer_epochs = 1000  # Adjust the number of epochs as needed

for epoch in range(customer_epochs):
    for i, row in df.iterrows():
        x = row[['Candies', 'Mangoes', 'Milk_Packets', 'Payment']].values  # Input features
        y_true = row['High_Value']  # True label

        # Calculate the weighted sum
        z = np.dot(x, weights[0:4]) + weights[4]

        # Calculate the predicted output using the sigmoid activation function
        y_pred = sigmoid(z)

        # Compute the error
        customer_error = y_true - y_pred

        # Update weights using gradient descent
        # weights += customer_learning_rate * customer_error * y_pred * (1 - y_pred) * x

# Function to predict the class (High or Low Value) based on the trained model
def predict(x):
    z = np.dot(x, weights[0:4]) + weights[4]
    y_pred = sigmoid(z)
    return 1 if y_pred > 0.5 else 0

# Test the model on new data
new_data = {
    'Candies': [23, 17, 30],
    'Mangoes': [5, 2, 4],
    'Milk_Packets': [3, 5, 3],
    'Payment': [1, 0, 1]  # Encoding Yes=1, No=0
}

df_new = pd.DataFrame(new_data)

# Make predictions
predictions = [predict(row[['Candies', 'Mangoes', 'Milk_Packets', 'Payment']].values) for _, row in df_new.iterrows()]

# Print predictions
for i, pred in enumerate(predictions):
    print(f"Data point {i + 1} is predicted as {'High Value' if pred == 1 else 'Low Value'}")

# Evaluation (you can use more advanced metrics like accuracy, precision, recall, etc.)
actual_labels = [1, 0, 1]  # Ground truth labels for the new data
accuracy = sum(predictions[i] == actual_labels[i] for i in range(len(predictions))) / len(predictions)
print(f"Accuracy: {accuracy * 100}%")