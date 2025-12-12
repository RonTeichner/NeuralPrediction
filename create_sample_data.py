import pandas as pd
import os

# Path to the full dataset
input_path = "/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/AmitShmidov/regressions_comparisons/spikes_and_metrics_dfs/dataset_somata.csv"

# Output path (save to repository's data directory)
output_dir = "data"
output_path = os.path.join(output_dir, "sample_dataset_somata.csv")

# Create data directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the full CSV file
print("Reading full dataset...")
df = pd.read_csv(input_path)#, sep='\t')

print(f"Full dataset shape: {df.shape}")
print(f"Unique experiments: {df['experiment'].nunique()}")
print(f"Unique channels: {df['channel'].nunique()}")

# Get two different experiments
experiments = df['experiment'].unique()[:2]
print(f"\nSelected experiments: {experiments}")

# Create sample dataset
sample_data = []

for exp in experiments:
    # Filter data for this experiment
    exp_data = df[df['experiment'] == exp]
    
    # Get two different channels from this experiment
    channels = exp_data['channel'].unique()[:2]
    print(f"Experiment {exp}: selected channels {channels}")
    
    for channel in channels:
        # Get all rows for this experiment and channel
        channel_data = exp_data[exp_data['channel'] == channel]
        sample_data.append(channel_data)

# Combine all sample data
sample_df = pd.concat(sample_data, ignore_index=True)

# Sort by time for better readability
sample_df = sample_df.sort_values('time').reset_index(drop=True)

print(f"\nSample dataset shape: {sample_df.shape}")
print(f"Sample contains:")
for exp in sample_df['experiment'].unique():
    exp_channels = sample_df[sample_df['experiment'] == exp]['channel'].unique()
    print(f"  Experiment {exp}: channels {exp_channels}")

# Save to CSV
sample_df.to_csv(output_path, index=False)#, sep='\t', index=False)
print(f"\nSample dataset saved to: {output_path}")
