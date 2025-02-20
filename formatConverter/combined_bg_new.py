import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

def calculate_qcd_ratio(frac_file_path, nevents_file_path, ht_bin):
    # Read fractions from .txt files
    fractions = {}
    
    with open(frac_file_path, 'r') as file:
        for index, line in enumerate(file):
            if index == 0:
                continue  # Skip the first line
            parts = line.strip().split()
            ht = float(parts[1])
            qcd_fraction = float(parts[2])
            tt_fraction = float(parts[3])
            W_fraction = float(parts[4])
            ST_fraction = float(parts[5])
            fractions[ht] = (qcd_fraction, tt_fraction, W_fraction, ST_fraction)

    # Read event numbers from the event numbers .txt file
    event_numbers = {}
    with open(nevents_file_path, 'r') as file:
        for index, line in enumerate(file):
            if index == 0:
                continue  # Skip the first line
            parts = line.strip().split()
            ht = float(parts[0])
            N_QCD_total = int(parts[1])
            N_TTbar_total = int(parts[2])
            N_WJets_total = int(parts[3])
            N_ST_total = int(parts[4])
            event_numbers[ht] = (N_QCD_total, N_TTbar_total, N_WJets_total, N_ST_total)

    qcd_fraction, tt_fraction, W_fraction, ST_fraction = fractions[ht_bin]
    N_QCD_total, N_TTbar_total, N_WJets_total, N_ST_total = event_numbers[ht_bin]
    N_selected_total = N_QCD_total
   

    qcd_ratio = 0
    TT_ratio = 0
    WJets_ratio = 0
    ST_ratio = 0

    if qcd_fraction == 0:
        qcd_fraction = 0.99
        tt_fraction = 0.005
        W_fraction = 0.005
        ST_fraction = 0

    if N_WJets_total ==N_ST_total==0:
        qcd_fraction = 0.99
        tt_fraction = 0.01
        W_fraction = 0
        ST_fraction = 0
    
    print(qcd_fraction,tt_fraction,W_fraction,ST_fraction)
    print(N_selected_total)
   
   
        # Construct the matrix and the vector
    A = np.array([
        [1 - ST_fraction, -ST_fraction, -ST_fraction],
        [-tt_fraction, 1 - tt_fraction, -tt_fraction],
        [-W_fraction, -W_fraction, 1 - W_fraction]
    ])
    b = np.array([
        ST_fraction * N_selected_total,
        tt_fraction * N_selected_total,
        W_fraction * N_selected_total
    ])

        # Solve the system of linear equations
    N_selected = np.linalg.solve(A, b)

    N_ST_selected = int(N_selected[0])
    N_TTbar_selected = int(N_selected[1])
    N_WJets_selected = int(N_selected[2])
    if N_ST_total == 0:
        ST_ratio =0
    else:
        ST_ratio = float(N_ST_selected) / N_ST_total
    TT_ratio = float(N_TTbar_selected) / N_TTbar_total
    if N_WJets_total == 0:
        WJets_ratio =0
    else:
        WJets_ratio = float(N_WJets_selected) / N_WJets_total
    qcd_ratio = 1

    if TT_ratio>1:
        TT_ratio = 1
    if ST_ratio >1:
        ST_ratio = 1
    if WJets_ratio > 1:
        WJets_ratio = 1

       
             
    return qcd_ratio, TT_ratio, ST_ratio, WJets_ratio
        

def main():
    frac_file_paths = "./txt_files/background_proporitons_h_totHT_1b_2015.txt"
    nevents_file_path = "./txt_files/2015_nevents.txt"
    ht_bins = [i for i in range(1700, 9900, 200)]
    qcd_file = './h5samples/combine/QCD_Sample_2015_BESTinputs_train_1.h5'
    tt_file = './h5samples/combine/Top_Sample_2015_BESTinputs_train_1.h5'
    st_file = './h5samples/combine/ST_Sample_2015_BESTinputs_train_1.h5'
    wjets_file = './h5samples/combine/WJets_Sample_2015_BESTinputs_train_1.h5'
    output_file_path = './h5samples/bg_2015.h5'
    ratio_path='ratios_2015.txt'
    
    combined_data = []
    with h5py.File(qcd_file, 'r') as qcd, \
            h5py.File(tt_file, 'r') as tt, \
            h5py.File(st_file, 'r') as st, \
            h5py.File(wjets_file, 'r') as wjets:

        qcd_data = qcd['BES_vars'][:]
        tt_data = tt['BES_vars'][:]
        st_data = st['BES_vars'][:]
        wjets_data = wjets['BES_vars'][:]
        
        for ht_bin in ht_bins:
        
            qcd_ratio, TT_ratio, ST_ratio, WJets_ratio = calculate_qcd_ratio(frac_file_paths, nevents_file_path, ht_bin)
            with open(ratio_path, 'a') as ratio_file:
                ratio_file.write("{},{},{},{},{}\n".format(ht_bin, qcd_ratio, TT_ratio, ST_ratio, WJets_ratio))
            ht_min = ht_bin-100
            ht_max = ht_bin + 100
        
            qcd_filtered = qcd_data[(qcd_data[:, 111] >= ht_min) & (qcd_data[:, 111] < ht_max)]
            tt_filtered = tt_data[(tt_data[:, 111] >= ht_min) & (tt_data[:, 111] < ht_max)]
            st_filtered = st_data[(st_data[:, 111] >= ht_min) & (st_data[:, 111] < ht_max)]
            wjets_filtered = wjets_data[(wjets_data[:, 111] >= ht_min) & (wjets_data[:, 111] < ht_max)]
            print(ht_bin, qcd_filtered.shape,tt_filtered.shape)
            if len(qcd_filtered) > 0:
                if qcd_ratio == 1:
                    combined_data.append(qcd_filtered)
                elif qcd_ratio > 0:
                    qcd_selected, _ = train_test_split(qcd_filtered, train_size=qcd_ratio, shuffle=True, random_state=42)
                    combined_data.append(qcd_selected)
                print(qcd_filtered.shape if qcd_ratio == 1 else qcd_selected.shape)

            if len(tt_filtered) > 0:
                if TT_ratio == 1:
                    combined_data.append(tt_filtered)
                elif TT_ratio > 0:
                    tt_selected, _ = train_test_split(tt_filtered, train_size=TT_ratio, shuffle=True, random_state=42)
                    combined_data.append(tt_selected)
                print(tt_filtered.shape if TT_ratio == 1 else tt_selected.shape)

            if len(st_filtered) > 0:
                if ST_ratio == 1:
                    combined_data.append(st_filtered)
                elif ST_ratio > 0:
                    st_selected, _ = train_test_split(st_filtered, train_size=ST_ratio, shuffle=True, random_state=42)
                    combined_data.append(st_selected)
                print(st_filtered.shape if ST_ratio == 1 else st_selected.shape)

            if len(wjets_filtered) > 0:
                if WJets_ratio == 1:
                    combined_data.append(wjets_filtered)
                elif WJets_ratio > 0:
                    wjets_selected, _ = train_test_split(wjets_filtered, train_size=WJets_ratio, shuffle=True, random_state=42)
                    combined_data.append(wjets_selected)
                print(wjets_filtered.shape if WJets_ratio == 1 else wjets_selected.shape)

    combined_data = np.vstack(combined_data)
    with h5py.File(output_file_path, 'w') as output_file:
        output_file.create_dataset('BES_vars', data=combined_data)

if __name__ == "__main__":
    main()



    
    

        