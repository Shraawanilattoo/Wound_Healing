import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load dataset (assuming it's in CSV format)
df = pd.read_excel("/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Healthy_wound_lactate.xlsx")

# Preprocessing
df.fillna(method='ffill', inplace=True)  # Forward fill missing values
scaler = MinMaxScaler()
df[['Time', '% wound closure', 'Lactate (mM)']] = scaler.fit_transform(df[['Time', '% wound closure', 'Lactate (mM)']])

# Split into input (X) and output (y)
X = df[['Time', '% wound closure']].values
y = df['Lactate (mM)'].values

# Reshape for LSTM (if using RNN)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, 2)),
    Dense(25, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Predict on test data
y_pred = model.predict(X_test)
