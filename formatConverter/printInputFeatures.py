import numpy as np
import h5py

def print_input_features(file_path, event_number):
    #with h5py.File(file_path, 'r') as f:
        # Check if the event number exists in the dataset
        #if str(event_number) not in f['BES_vars']:
        #    print("Event number %s not found in the dataset."%(event_number))
        #    return
        #print("looking for event number")
        
    scaled_events = np.array(h5py.File(file_path,"r")["BES_vars"])[()]

    for superjet in scaled_events:
        print(superjet[178])
        if ( abs(event_number - int(superjet[178])) < 1e-5):
            print("found the event!") ## why are these events not found???
            print(superjet)

    #print(len(scaled_events[0]))

# Example usage:
file_path = 'h5samples/QCD_Sample_all_mass_2018_BESTinputs_test_flattened_standardized.h5'
event_number = 942198  # Specify the event number you want to find
print_input_features(file_path, event_number)
