import pandas as pd
import numpy as np
from scipy import stats

# Step 1: Load the dataset
df = pd.read_csv("your_data.csv")

# Step 2: Handle missing values
# Option 1: Drop rows with any missing values
df_cleaned = df.dropna()

# Option 2: Fill missing values with a specific value (mean, median, mode)
df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# Step 3: Remove duplicates
df_cleaned = df_cleaned.drop_duplicates()

# Step 4: Handle outliers (Using Z-Score or IQR)

# Using Z-Score method
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df_cleaned = df_cleaned[(z_scores < 3).all(axis=1)]  # Keep rows with Z-score < 3

# Alternatively, using IQR (Interquartile Range)
Q1 = df_cleaned.quantile(0.25)
Q3 = df_cleaned.quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df_cleaned[~((df_cleaned < (Q1 - 1.5 * IQR)) | (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]

# Step 5: Convert data types if necessary
df_cleaned['column_name'] = df_cleaned['column_name'].astype('int')

# Step 6: Standardize column names (lowercase and replace spaces with underscores)
df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')

# Step 7: Save the cleaned dataset
df_cleaned.to_csv("cleaned_data.csv", index=False)

# Optionally, view the cleaned data
print(df_cleaned.head())
