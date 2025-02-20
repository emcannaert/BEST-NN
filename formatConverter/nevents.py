import h5py
import numpy as np
import os

def get_ht_values(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['BES_vars'][:]  
        ht_values = data[:, -2]  # Assuming HT is the second last column
    return ht_values

def count_events_in_bins(ht_values, bins):
    counts, _ = np.histogram(ht_values, bins=bins)
    return counts

def main():
    ht_bins = np.arange(1600, 10000, 200)  # Define your HT bins
    all_counts = np.zeros(len(ht_bins) - 1)
    output_file = "ht_bin_counts.txt"

    with open(output_file, 'w') as f:

        for file_name in os.listdir('./h5samples'):
            if (file_name.endswith('.h5') and file_name.startswith('allDecay')==False):
                file_name = os.path.join("./h5samples/", file_name) 
                ht_values = get_ht_values(file_name)
                counts = count_events_in_bins(ht_values, ht_bins)
                all_counts += counts
                f.write("File: {}\n".format(file_name))
                for i in range(len(ht_bins) - 1):
                    f.write("  HT bin {}-{}: {} events\n".format(ht_bins[i], ht_bins[i+1], counts[i]))
            


if __name__ == "__main__":
    main()