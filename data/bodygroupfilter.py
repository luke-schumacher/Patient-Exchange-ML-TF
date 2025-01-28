import os
import pandas as pd

# Define the directory containing the CSV files
input_directory = "data/tokenization_182625"

# Loop through all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Check if the file contains the required condition: BodyGroup_from = 1 (Abdomen) and BodyGroup_to = 3 (Head) in the same row
        filtered_df = df[(df['BodyGroup_from'] == 6) & (df['BodyGroup_to'] == 6)]

        # If the filtered DataFrame is empty, remove the file
        if filtered_df.empty:
            os.remove(file_path)
            print(f"Removed file: {filename}")
        else:
            print(f"Kept file: {filename}")

print("Filtering complete.")
