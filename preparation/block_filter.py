import os
import pandas as pd

# Load your dataset (adjust the path to your data file)
data = pd.read_csv('data/encoded_175651.csv')

# Create a folder to store output CSVs
output_folder = 'filtered_blocks_175651'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize variables
start_idx = None
block_count = 1

# Iterate over the rows to find blocks between '10' and '9'
for idx, row in data.iterrows():
    if row['sourceID_encoded'] == 10:
        start_idx = idx  # Mark the start of a block
    elif row['sourceID_encoded'] == 9 and start_idx is not None:
        # Calculate the number of rows between '10' and '9'
        row_diff = idx - start_idx
        # Save blocks with 5 to 40 rows between '10' and '9'
        if 5 <= row_diff <= 40:
            block_data = data.loc[start_idx:idx+1]  # Extract the block from '10' to '9' (inclusive)
            block_data = block_data[:-1]  # Remove the last row (which might contain an extra '10')
            # Save block to CSV
            block_data.to_csv(f'{output_folder}/block_{block_count}.csv', index=False)
            block_count += 1
        start_idx = None  # Reset the start index for the next block

print(f'{block_count - 1} blocks saved to {output_folder}')
