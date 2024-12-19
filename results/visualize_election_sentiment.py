
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Load the datasets
file1_path = 'election_scores.csv'  # Update with your file path
file2_path = 'state_scores_roberta_model.csv'  # Update with your file path
file3_path = 'state_scores_georgetown_model.csv'  # Update with your file path

# Read the CSV files
election_scores_df = pd.read_csv(file1_path)
roberta_model_df = pd.read_csv(file2_path)
georgetown_model_df = pd.read_csv(file3_path)

# Merge the datasets on the 'state' column for analysis
roberta_merged_df = pd.merge(election_scores_df, roberta_model_df, on='state', suffixes=('_election', '_roberta'))
georgetown_merged_df = pd.merge(election_scores_df, georgetown_model_df, on='state', suffixes=('_election', '_georgetown'))

# Function to calculate slope, intercept, and corrected R-squared
def calculate_slope_intercept_r2_fixed(df, x_col, y_col):
    slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
    predicted = slope * df[x_col] + intercept
    r_squared = r2_score(df[y_col], predicted)  # Correct R-squared calculation
    return slope, intercept, r_squared

# Prepare subplots for 6 different comparisons
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Define model names and their respective column pairs
models = [
    ('roberta', 'average_avg_stance'),
    ('roberta', 'average_forced_stance'),
    ('roberta', 'weighted_avg_forced_stance'),
    ('georgetown', 'average_avg_stance'),
    ('georgetown', 'average_forced_stance'),
    ('georgetown', 'weighted_avg_forced_stance'),
]

# Iterate through models and plot
for i, (model, col) in enumerate(models):
    # Select the appropriate dataframe
    df = roberta_merged_df if model == 'roberta' else georgetown_merged_df
    ax = axes[i // 3, i % 3]
    
    # Calculate slope, intercept, and corrected R-squared
    slope, intercept, r_squared = calculate_slope_intercept_r2_fixed(df, 'score', col)
    
    # Scatter plot and linear fit
    ax.scatter(df['score'], df[col], label=f'{model.capitalize()} {col.replace("_", " ").capitalize()}')
    ax.plot(df['score'], slope * df['score'] + intercept, color='red', label=f'Fit (R²={r_squared:.2f}, Slope={slope:.2f})')
    
    # Formatting the plot
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.7)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.7)
    ax.set_title(f'{model.capitalize()} {col.replace("_", " ").capitalize()} vs Actual Results')
    ax.set_xlabel('Actual Election Scores')
    ax.set_ylabel('Predicted Scores')
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()
