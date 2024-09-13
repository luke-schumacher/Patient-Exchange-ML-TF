import pandas as pd

# Define file paths directly relative to the current working directory (PatientExchange)
input_file = "data/mr75609filt.csv"
output_file = "data/encoded_mr75609filt_version4.csv"

# Load the data
df = pd.read_csv(input_file)

# Print the column names to check if 'BodyPart' and 'sourceID' exist
print("Columns in the DataFrame:", df.columns)

# Generate the legends using original text values
bodypart_legend = df['BodyPart'].astype('category').cat.categories
sourceid_legend = df['sourceID'].astype('category').cat.categories

# Map the original text names to numeric codes
df['BodyPart'] = df['BodyPart'].astype('category').cat.codes + 1  # Shift codes to be positive
df['sourceID'] = df['sourceID'].astype('category').cat.codes + 1  # Shift codes to be positive

# Drop the 'text' column as it's not needed
df = df.drop(columns=['text'])

# Save the encoded data to a new CSV file
df.to_csv(output_file, index=False)

# Print the legends for reference
print("BodyPart Encoding Legend:")
for code, bodypart in enumerate(bodypart_legend, start=1):
    print(f"{code}: {bodypart}")

print("\nsourceID Encoding Legend:")
for code, sourceid in enumerate(sourceid_legend, start=1):
    print(f"{code}: {sourceid}")
