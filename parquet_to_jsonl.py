import pandas as pd

# Specify the list of Parquet file paths using raw strings
parquet_files = [
    r"parquet_data/hindi0.parquet",
    r"parquet_data/hindi1.parquet",
    r"parquet_data/hindi2.parquet",
    r"parquet_data/hindi3.parquet",
    r"parquet_data/hindi4.parquet",
    r"parquet_data/marathi0.parquet",
    r"parquet_data/marathi1.parquet",
    r"parquet_data/marathi2.parquet",
    r"parquet_data/marathi3.parquet",
    r"parquet_data/marathi4.parquet",
    r"parquet_data/panjabi0.parquet",
    r"parquet_data/panjabi1.parquet",
    r"parquet_data/panjabi2.parquet",
    r"parquet_data/panjabi3.parquet",
    r"parquet_data/panjabi4.parquet"
]

# Initialize an empty list to store DataFrames
dfs = []

# Read each Parquet file, print its shape, and append to the list
for file in parquet_files:
    df = pd.read_parquet(file)
    print(f"Shape of {file}: {df.shape}")
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Convert to JSONL and save
jsonl_file = "final_combined_data.jsonl"
combined_df.to_json(jsonl_file, orient='records', lines=True)

# Print the shape of the combined DataFrame and confirmation
print(f"\nSuccessfully converted {len(parquet_files)} Parquet files to {jsonl_file}")
print(f"Shape of combined DataFrame: {combined_df.shape}")