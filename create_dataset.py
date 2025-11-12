
import pandas as pd
import numpy as np
import os

# Get the absolute path of the current working directory
# Note: This script assumes it is being run from the project's root directory.
# If running from a different directory, you may need to adjust the base_path.
base_path = os.getcwd()

# Construct absolute paths for the files
positive_file_path = os.path.join(base_path, 'nct_labelled.csv')
negative_file_path = os.path.join(base_path, 'Network Change Tickets (NCT).xlsx')
output_file_path = os.path.join(base_path, 'training_dataset.csv')


# Load the positive examples
positive_df = pd.read_csv(positive_file_path)

# Add the target label
positive_df['sred_eligible'] = 1

# Load the negative examples from the '2020' sheet
negative_df_source = pd.read_excel(negative_file_path, sheet_name='2020')

# Get the 'Nct Id's from the positive examples to avoid data contamination
positive_nct_ids = positive_df['Nct Id'].unique()

# Filter out the positive examples from the master list to create a negative pool
negative_pool_df = negative_df_source[~negative_df_source['Nct Id'].isin(positive_nct_ids)].copy()

# Determine the number of negative samples to take (2x the number of positives)
num_positive_samples = len(positive_df)
num_negative_samples = num_positive_samples * 2

# Randomly sample from the negative pool
# Ensure we don't try to sample more than we have
if len(negative_pool_df) < num_negative_samples:
    num_negative_samples = len(negative_pool_df)

negative_samples_df = negative_pool_df.sample(n=num_negative_samples, random_state=42)

# Add the target label to the negative samples
negative_samples_df['sred_eligible'] = 0

# Identify common columns to ensure a clean merge
common_columns = list(set(positive_df.columns) & set(negative_samples_df.columns))

# Filter both dataframes to only have common columns plus the new target column
positive_df_aligned = positive_df[common_columns + ['sred_eligible']]
negative_samples_df_aligned = negative_samples_df[common_columns + ['sred_eligible']]


# Combine the positive and negative samples
final_dataset = pd.concat([positive_df_aligned, negative_samples_df_aligned], ignore_index=True)

# Save the final dataset to a CSV file
final_dataset.to_csv(output_file_path, index=False)

print(f"Dataset created successfully with {len(final_dataset)} records.")
print(f"Positive samples: {len(positive_df)}")
print(f"Negative samples: {len(negative_samples_df)}")
print(f"Dataset saved to: {output_file_path}")
