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
        new_column = np.full((modified_data.shape[0], 1), 3, dtype=int)
        modified_data = np.hstack((modified_data, new_column))
        
        #add one number at the end of the data to record their type. 0 is QCD, 1 is TT, 2 is WJets, 3 is ST
       
        # Delete the original dataset
        del f[dataset_name]
        # Create a new dataset with the modified data
        f.create_dataset(dataset_name, data=modified_data)

# Usage
h5_file = 'ST_Sample_2018_BESTinputs.h5'
dataset_name = 'BES_vars'
columns_to_delete = [112]  

# Delete specified columns
delete_columns(h5_file, dataset_name, columns_to_delete)


# Verify the deletion
bes_vars_array, size = get_bes_vars_as_array(h5_file, dataset_name)
print("Dataset size after deletion:", size)
print(bes_vars_array)