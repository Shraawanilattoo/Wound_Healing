import pandas as pd

# Load the Excel file
file_path = "Diabetic_wound_lactate.xlsx"  # Update the path if needed
xls = pd.ExcelFile(file_path)

# Load the first sheet
df = pd.read_excel(xls, sheet_name="Sheet1")

# Display the first few rows
print(df)

# Number of diabetic animals (assuming 4 based on column structure)
num_animals = 4  

# List to store structured data
structured_data = []

# Process each animal’s data
for i in range(num_animals):
    base_col = i * 4  # Each animal's data spans 4 columns
    time_col = base_col
    lactate_col = base_col + 1
    wound_closure_col = base_col + 2
    
    # Extract non-null values
    time_data = df.iloc[2:, time_col].dropna().values  # Time (Days)
    lactate_data = df.iloc[2:, lactate_col].dropna().values  # Lactate (mM)
    wound_closure_data = df.iloc[2:, wound_closure_col].fillna(0).values  # Wound Closure (%), fill missing values with 0
    
    # Ensure consistent length
    min_length = min(len(time_data), len(lactate_data), len(wound_closure_data))
    
    for j in range(min_length):
        structured_data.append([time_data[j], lactate_data[j], wound_closure_data[j]])

# Convert structured data into a DataFrame
cleaned_df = pd.DataFrame(structured_data, columns=["Time (Days)", "Lactate (mM)", "% Wound Closure"])

# Display the cleaned dataset
print(cleaned_df)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Select features (X) and target (y)
X = cleaned_df[["Time (Days)", "Lactate (mM)"]].values  # Inputs
y = cleaned_df["% Wound Closure"].values  # Target

# Normalize the input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
