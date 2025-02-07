import h5py
import numpy as np

def get_bes_vars_as_array(h5_file, dataset_name):
    with h5py.File(h5_file, 'r') as f:
        data = np.array(f[dataset_name])
        size = f[dataset_name].shape
    return data, size

def delete_columns(h5_file, dataset_name, columns_to_delete):
    with h5py.File(h5_file, 'a') as f:
        data = np.array(f[dataset_name])
        # Delete the specified columns
        modified_data = np.delete(data, columns_to_delete, axis=1)
        
        #add one number at the end of the data to record their type. 0 is QCD, 1 is TT, 2 is WJets, 3 is ST
        new_column = np.ones((modified_data.shape[0], 1), dtype=int) 
        modified_data = np.hstack((modified_data, new_column))
        # Delete the original dataset
        del f[dataset_name]
        # Create a new dataset with the modified data
        f.create_dataset(dataset_name, data=modified_data)

# Usage
h5_file = 'Top_Sample_all_mass_combine_BESTinputs_validation.h5'
dataset_name = 'BES_vars'
columns_to_delete = [7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 141, 142, 143, 144, 145, 146, 147, 148, 149, 180]  

# Delete specified columns
delete_columns(h5_file, dataset_name, columns_to_delete)


# Verify the deletion
bes_vars_array, size = get_bes_vars_as_array(h5_file, dataset_name)
print("Dataset size after deletion:", size)
print(bes_vars_array)