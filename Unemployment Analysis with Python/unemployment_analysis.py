import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the unemployment dataset
# Replace 'unemployment_data.csv' with your actual dataset filename
df = pd.read_csv('unemployment_data.csv')

# Basic data exploration
print(df.head())
print(df.describe())
print(df.info())

# Plot unemployment rate over time
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x='Date', y='Unemployment Rate')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
