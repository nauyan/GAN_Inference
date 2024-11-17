import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

# Load the original CSV file
train_data = pd.read_csv("Credit.csv")

# Select only the 'Amount' column
amount_data = train_data[["Amount"]]

# Determine the number of rows in the original data
num_original_rows = len(amount_data)

# Define the target number of rows
target_rows = 1000000000
shard_size = 10000000

# Calculate the number of times the original data needs to be repeated
repeat_count = target_rows // num_original_rows + 1

# Repeat the data for the 'Amount' column
repeated_amount_data = pd.concat([amount_data] * repeat_count, ignore_index=True)

# Truncate the DataFrame to exactly 1 billion rows
repeated_amount_data = repeated_amount_data.iloc[:target_rows]

# Prepare output directory
os.makedirs("data", exist_ok=True)


# Helper function to save a shard
def save_shard(shard_index):
    start_row = shard_index * shard_size
    end_row = start_row + shard_size
    shard = repeated_amount_data.iloc[start_row:end_row]

    # Save as CSV
    shard_csv_path = f"data/Amount_Shard_{shard_index + 1}.csv"
    shard.to_csv(shard_csv_path, index=False)

    # Save as Parquet
    shard_parquet_path = f"data/Amount_Shard_{shard_index + 1}.parquet"
    shard.to_parquet(shard_parquet_path, index=False, engine="pyarrow")

    return f"Shard {shard_index + 1} saved: CSV -> {shard_csv_path}, Parquet -> {shard_parquet_path}"


# Total number of shards
num_shards = target_rows // shard_size

# Use multiprocessing to parallelize shard saving
if __name__ == "__main__":
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(
            pool.imap_unordered(save_shard, range(num_shards)), total=num_shards
        ):
            print(result)
