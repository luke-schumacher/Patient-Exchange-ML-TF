import csv

# Read the input file
with open('results.txt', 'r') as file:
    data = file.read()

# Extract MRI-related entries and ignore START/END
lines = data.splitlines()
mri_entries = []
for line in lines:
    # Extract everything between START and END, split, and filter
    content = line.split(': ')[-1].strip("'")
    entries = [item for item in content.split() if item.startswith('MRI_')]
    mri_entries.extend(entries)

# Write to a CSV file
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['SourceID'])  # Write the column header
    for entry in mri_entries:
        writer.writerow([entry])

print("CSV file created: results.csv")
