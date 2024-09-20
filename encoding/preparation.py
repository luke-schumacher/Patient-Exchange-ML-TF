import pandas as pd

# Define file paths directly relative to the current working directory (PatientExchange)
input_file = "Patient-Exchange-ML-TF/data/filt_176398_Brain.csv"
output_file = "Patient-Exchange-ML-TF/data/encoded_176398_Brain.csv"

# Load the data
df = pd.read_csv(input_file)

# Print the column names to check if 'BodyGroup_to', 'BodyGroup_from', and 'sourceID' exist
print("Columns in the DataFrame:", df.columns)

# Generate the legends using original text values for 'BodyGroup_to', 'BodyGroup_from', and 'sourceID'
bodygroup_to_legend = df['BodyGroup_to'].astype('category').cat.categories
bodygroup_from_legend = df['BodyGroup_from'].astype('category').cat.categories
sourceid_legend = df['sourceID'].astype('category').cat.categories

# Map the original text names to numeric codes for 'BodyGroup_to', 'BodyGroup_from', and 'sourceID'
df['BodyGroup_to'] = df['BodyGroup_to'].astype('category').cat.codes + 1  # Shift codes to be positive
df['BodyGroup_from'] = df['BodyGroup_from'].astype('category').cat.codes + 1  # Shift codes to be positive
df['sourceID'] = df['sourceID'].astype('category').cat.codes + 1  # Shift codes to be positive

# Drop the 'text' column as it's not needed
df = df.drop(columns=['text'])

# Save the encoded data to a new CSV file
df.to_csv(output_file, index=False)

# Print the legends for reference
print("BodyGroup_to Encoding Legend:")
for code, bodygroup_to in enumerate(bodygroup_to_legend, start=1):
    print(f"{code}: {bodygroup_to}")

print("\nBodyGroup_from Encoding Legend:")
for code, bodygroup_from in enumerate(bodygroup_from_legend, start=1):
    print(f"{code}: {bodygroup_from}")

print("\nsourceID Encoding Legend:")
for code, sourceid in enumerate(sourceid_legend, start=1):
    print(f"{code}: {sourceid}")
