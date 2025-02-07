import h5py
import numpy as np
import os

def get_ht_values(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['BES_vars'][:]  # Replace 'dataset_name' with the actual dataset name
        ht_values = data[:, -2]  # Assuming HT is the second last column
    return ht_values

def count_events_in_bins(ht_values, bins):
    counts, _ = np.histogram(ht_values, bins=bins)
    return counts

def main():
    ht_bins = np.arange(100, 10000, 200)  # Define your HT bins
    all_counts = np.zeros(len(ht_bins) - 1)
    output_file = "ht_bin_counts.txt"

    with open(output_file, 'w') as f:

        for file_name in os.listdir('.'):
            if file_name.endswith('.h5'):
                ht_values = get_ht_values(file_name)
                counts = count_events_in_bins(ht_values, ht_bins)
                all_counts += counts
                f.write(f"File: {file_name}\n")
                for i in range(len(ht_bins) - 1):
                    f.write(f"  HT bin {ht_bins[i]}-{ht_bins[i+1]}: {counts[i]} events\n")
            


if __name__ == "__main__":
    main()