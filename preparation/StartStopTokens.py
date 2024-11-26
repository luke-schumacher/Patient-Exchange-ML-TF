import os
import pandas as pd

# Directory containing the original CSVs
input_dir = './data/filtered_blocks_padded/'  # Replace with the directory containing your CSV files
output_dir = './tokenization'  # Directory to save the processed CSVs

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all CSV files in the directory
csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

print(f"Found {len(csv_files)} CSV files in directory: {input_dir}")

# Process each CSV file
for csv_file in csv_files:
    file_path = os.path.join(input_dir, csv_file)
    print(f"Processing file: {csv_file}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    print(f"Initial shape of {csv_file}: {df.shape}")

    # Ensure required columns exist
    required_columns = ["sourceID", "timediff", "PTAB", "BodyGroup_from", "BodyGroup_to"]
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in {csv_file}")
            continue

    # Add 13 row with updated values
    first_values = {
        "timediff": 0,
        "PTAB": df["PTAB"].dropna().iloc[0] if not df["PTAB"].dropna().empty else None,
        "BodyGroup_from": df["BodyGroup_from"].dropna().iloc[0] if not df["BodyGroup_from"].dropna().empty else None,
        "BodyGroup_to": df["BodyGroup_to"].dropna().iloc[0] if not df["BodyGroup_to"].dropna().empty else None,
    }
    new_row_13 = pd.DataFrame([{"sourceID": 13, **first_values}])
    df = pd.concat([new_row_13, df]).reset_index(drop=True)

    # Check if sourceID 9 exists before adding 14
    if 9 in df["sourceID"].values:
        last_values = {
            "timediff": df["timediff"].dropna().iloc[-1] if not df["timediff"].dropna().empty else None,
            "PTAB": df["PTAB"].dropna().iloc[-1] if not df["PTAB"].dropna().empty else None,
            "BodyGroup_from": df["BodyGroup_from"].dropna().iloc[-1] if not df["BodyGroup_from"].dropna().empty else None,
            "BodyGroup_to": df["BodyGroup_to"].dropna().iloc[-1] if not df["BodyGroup_to"].dropna().empty else None,
        }
        new_row_14 = pd.DataFrame([{"sourceID": 14, **last_values}])
        index_9 = df.index[df["sourceID"] == 9][0]
        df = pd.concat([df.iloc[:index_9 + 1], new_row_14, df.iloc[index_9 + 1:]]).reset_index(drop=True)
    else:
        print(f"No sourceID 9 found in {csv_file}, skipping addition of 14.")

    # Remove exactly two occurrences of 0 in the sourceID column
    zeros_indices = df.index[df["sourceID"] == 0]
    if len(zeros_indices) >= 2:
        df = df.drop(zeros_indices[:2]).reset_index(drop=True)

    # Replace all remaining 0s in sourceID with 14
    df["sourceID"] = df["sourceID"].replace(0, 14)

    print(f"Modified shape of {csv_file}: {df.shape}")

    # Save the processed DataFrame to the output directory
    output_path = os.path.join(output_dir, csv_file)
    df.to_csv(output_path, index=False)
    print(f"Processed file saved to: {output_path}")

print("Processing completed.")
