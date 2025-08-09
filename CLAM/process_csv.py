import pandas as pd
import os

# Paths
input_csv = "/mnt/e/challenge/github/REG_histgen/CLAM/test2_patches/process_list_autogen.csv"
output_csv = "/mnt/e/challenge/github/REG_histgen/CLAM/test2_patches/df_features.csv"

# Read CSV
df = pd.read_csv(input_csv)

# Remove '.tiff' from slide_id
df['slide_id'] = df['slide_id'].str.replace('.tiff', '', regex=False)

# Save new CSV
df.to_csv(output_csv, index=False)

print(f"Updated CSV saved to: {output_csv}")