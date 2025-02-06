import pandas as pd

# Define file paths
data_file = "data/filtered_175651.csv"
output_file = "data/encoded_175651.csv"

# Encoding legends
ENCODING_LEGEND = {
    'Not Vital': 0, 'MRI_CCS_11': 1, 'MRI_EXU_95': 2, 'MRI_FRR_18': 3, 'MRI_FRR_257': 4,
    'MRI_FRR_264': 5, 'MRI_FRR_3': 6, 'MRI_FRR_34': 7, 'MRI_MPT_1005': 8,
    'MRI_MSR_100': 9, 'MRI_MSR_104': 10, 'MRI_MSR_21': 11, 'MRI_MSR_34': 12
}

BODYGROUP_ENCODING = {
    'ABDOMEN': 1, 'ARM': 2, 'HEAD': 3, 'HEART': 4, 'HIP': 5,
    'KNEE': 6, 'LEG': 7, 'PELVIS': 8, 'SHOULDER': 9, 'SPINE': 10
}

# Load data
df = pd.read_csv(data_file)

# Print column names to verify required columns exist
print("Columns in the DataFrame:", df.columns.tolist())

# Encode 'sourceID' using predefined legend
df['sourceID_encoded'] = df['sourceID'].map(ENCODING_LEGEND)

# Encode 'BodyGroup_to' and 'BodyGroup_from' using predefined legend
df['BodyGroup_to_encoded'] = df['BodyGroup_to'].map(BODYGROUP_ENCODING)
df['BodyGroup_from_encoded'] = df['BodyGroup_from'].map(BODYGROUP_ENCODING)

# Handle missing mappings (if any values were not found in the legends)
df.fillna({'sourceID_encoded': 0, 'BodyGroup_to_encoded': 0, 'BodyGroup_from_encoded': 0}, inplace=True)
df = df.astype({'sourceID_encoded': 'int32', 'BodyGroup_to_encoded': 'int32', 'BodyGroup_from_encoded': 'int32'})

# Save the encoded data
df_encoded = df.drop(columns=['BodyGroup_to', 'BodyGroup_from', 'sourceID'])
df_encoded.to_csv(output_file, index=False)

# Print encoding legends
print("\nsourceID Encoding Legend:")
for key, value in ENCODING_LEGEND.items():
    print(f"{value}: {key}")

print("\nBodyGroup Encoding Legend:")
for key, value in BODYGROUP_ENCODING.items():
    print(f"{value}: {key}")

# Optional: Display original vs encoded comparison
print("\nOriginal vs Encoded Comparison:")
comparison_df = df[['BodyGroup_to', 'BodyGroup_to_encoded', 
                    'BodyGroup_from', 'BodyGroup_from_encoded', 
                    'sourceID', 'sourceID_encoded']].head()
print(comparison_df)
