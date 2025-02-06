import os
import pandas as pd
import numpy as np

def pad_csv_files(data_dir, output_dir, target_length=36):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List of columns to drop
    columns_to_drop = [
        'datetime', 'SN', 'ZAxisInPossible', 'ZAxisOutPossible', 'YAxisDownPossible',
        'YAxisUpPossible', 'BC', 'S1', 'S10', 'S11', 'S12', 'S2', 'S3', 'S4',
        'S5', 'S6', 'S7', 'S8', 'S9', 'BO1', 'BO2', 'BO3', 'B1', 'B2', 'B3', 'B4',
        'B5', 'HE2', 'HE4', 'NE2', 'HE1', 'HE3', 'NE1', 'SHA', 'HW1', 'HW2', 'HW3',
        '18K', 'FA', 'TO', 'BAL', 'BAR', 'BCL', 'BCR', 'HC2', 'HC4', 'HC6', 'HC7',
        'NC2', 'HC1', 'HC3', 'HC5', 'NC1', 'Na', 'UFL', 'PA1', 'PA2', 'PA3', 'PA4',
        'PA5', 'PA6', 'SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'SP6', 'SP7', 'SP8', 'BL8',
        'BR8', 'UFS', 'HEA', 'HEP', 'SC', 'PeH', 'PeN', 'FS', 'FL', 'BY1', 'BY2',
        'BY3', 'BL', 'BR', 'HE', 'BL4', 'BR4', 'BL1', 'BR1', 'BL2', 'BR2', 'L7',
        'L4', 'H2L', 'N2L', 'H1U', 'N1U', 'He1', 'He2', 'TR1', 'TR2', 'TR3', 'TR4',
        'TR5', 'TR6', 'MR', 'ML', 'BL5', 'BR5', 'C24', 'EN', 'SHL', 'SHS', 'BodyPart_from',
        'BodyPart_to', 'PatientID_from', 'PatientID_to'
    ]

    # Columns to be padded with the last seen value
    columns_to_pad = ['PTAB', 'timediff', 'BodyGroup_from', 'BodyGroup_to']

    # List to store dataframes
    padded_dfs = []

    # Read all CSV files and pad them to the target length
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)

            # Drop specified columns
            df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

            # Calculate how much padding is needed
            padding_needed = target_length - len(df)

            if padding_needed > 0:
                # Create padding DataFrame with the last value for specific columns
                padding_dict = {}
                for col in columns_to_pad:
                    if col in df.columns:
                        last_value = df[col].iloc[-1]  # Get the last seen value for the column
                        padding_dict[col] = [last_value] * padding_needed  # Repeat last value

                padding_df = pd.DataFrame(padding_dict)

                # Add zeros to the remaining columns that are not in columns_to_pad
                for col in df.columns:
                    if col not in columns_to_pad:
                        padding_df[col] = 0

                # Concatenate the original df with the padding
                padded_df = pd.concat([df, padding_df], ignore_index=True)
            else:
                # If df is already longer or equal to target length, slice it
                padded_df = df.iloc[:target_length].reset_index(drop=True)

            padded_dfs.append(padded_df)

            # Save the padded DataFrame to the output directory
            padded_df.to_csv(os.path.join(output_dir, f'padded_{os.path.basename(filename)}'), index=False)

    # Print the lengths of all padded DataFrames to confirm homogeneity
    print("Lengths of padded DataFrames:")
    for padded_df in padded_dfs:
        print(len(padded_df))


# Usage
data_directory = 'filtered_blocks_175651'  # Replace with your CSV directory path
output_directory = 'filtered_blocks_padded_175651'  # Replace with your desired output directory
pad_csv_files(data_directory, output_directory)
