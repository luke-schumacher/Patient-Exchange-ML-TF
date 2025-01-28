import pandas as pd

# Define file paths directly relative to the current working directory (PatientExchange)
input_file = "data/filtered_182625.csv"
output_file = "data/encoded_182625.csv"

# Load the data
df = pd.read_csv(input_file)

# Print the column names to check if 'BodyGroup_to', 'BodyGroup_from', and 'sourceID' exist
print("Columns in the DataFrame:", df.columns)

# Generate the legends using original text values for 'BodyGroup_to', 'BodyGroup_from', and 'sourceID'
bodygroup_to_legend = df['BodyGroup_to'].astype('category').cat.categories
bodygroup_from_legend = df['BodyGroup_from'].astype('category').cat.categories
sourceid_legend = df['sourceID'].astype('category').cat.categories

# Map the original text names to numeric codes for 'BodyGroup_to', 'BodyGroup_from', and 'sourceID'
df['BodyGroup_to_encoded'] = df['BodyGroup_to'].astype('category').cat.codes + 1  # Shift codes to be positive
df['BodyGroup_from_encoded'] = df['BodyGroup_from'].astype('category').cat.codes + 1  # Shift codes to be positive
df['sourceID_encoded'] = df['sourceID'].astype('category').cat.codes + 1  # Shift codes to be positive

# Save the encoded data to a new CSV file
df_encoded = df.drop(columns=['BodyGroup_to', 'BodyGroup_from', 'sourceID'])  # Keep only encoded versions
df_encoded.to_csv(output_file, index=False)

# Print the legends for reference
print("\nBodyGroup_to Encoding Legend:")
for code, bodygroup_to in enumerate(bodygroup_to_legend, start=1):
    print(f"{code}: {bodygroup_to}")

print("\nBodyGroup_from Encoding Legend:")
for code, bodygroup_from in enumerate(bodygroup_from_legend, start=1):
    print(f"{code}: {bodygroup_from}")

print("\nsourceID Encoding Legend:")
for code, sourceid in enumerate(sourceid_legend, start=1):
    print(f"{code}: {sourceid}")

# Optional: Display the first few rows of the original and encoded columns for comparison
print("\nOriginal vs Encoded Comparison:")
comparison_df = df[['BodyGroup_to', 'BodyGroup_to_encoded', 
                    'BodyGroup_from', 'BodyGroup_from_encoded', 
                    'sourceID', 'sourceID_encoded']].head()
print(comparison_df)
