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
    
    

     # Use the smallest number of events (N_ST_total) to determine the selected events
    if N_ST_total == 0 and N_WJets_total !=0:
        N_selected_total = N_WJets_total
    elif N_ST_total != 0 and N_WJets_total ==0:
        N_selected_total = N_ST_total
    elif N_ST_total == 0 and N_WJets_total ==0:
        N_selected_total = N_QCD_total
    else:
        N_selected_total = min(N_ST_total, N_WJets_total)

    qcd_ratio = 0
    TT_ratio = 0
    WJets_ratio = 0
    ST_ratio = 0

    if qcd_fraction == 0:
        qcd_fraction = 0.99
        tt_fraction = 0.005
        W_fraction = 0.005
        ST_fraction = 0
    
    print(qcd_fraction,tt_fraction,W_fraction,ST_fraction)
    print(N_selected_total)
   
    if (N_selected_total == N_ST_total) and (ST_fraction != 0):
        # Construct the matrix and the vector
        A = np.array([
            [1 - qcd_fraction, -qcd_fraction, -qcd_fraction],
            [-tt_fraction, 1 - tt_fraction, -tt_fraction],
            [-W_fraction, -W_fraction, 1 - W_fraction]
        ])
        b = np.array([
            qcd_fraction * N_selected_total,
            tt_fraction * N_selected_total,
            W_fraction * N_selected_total
        ])

        # Solve the system of linear equations
        N_selected = np.linalg.solve(A, b)

        N_QCD_selected = int(N_selected[0])
        N_TTbar_selected = int(N_selected[1])
        N_WJets_selected = int(N_selected[2])
        qcd_ratio = float(N_QCD_selected) / N_QCD_total
        TT_ratio = float(N_TTbar_selected) / N_TTbar_total
        if N_WJets_total == 0:
            WJets_ratio =0
        else:
            WJets_ratio = float(N_WJets_selected) / N_WJets_total
        ST_ratio = 1

        if (qcd_ratio >1):
            N_selected_total = N_QCD_total
            A = np.array([
                [1 - tt_fraction, -tt_fraction, -tt_fraction],
                [-ST_fraction, 1 - ST_fraction, -ST_fraction],
                [-W_fraction, -W_fraction, 1 - W_fraction]
              ])
            b = np.array([
                tt_fraction * N_selected_total,
                ST_fraction * N_selected_total,
                W_fraction * N_selected_total
                ])

            # Solve the system of linear equations
            N_selected = np.linalg.solve(A, b)

            N_WJets_selected = int(N_selected[2])
            N_TTbar_selected = int(N_selected[0])
            N_ST_selected = int(N_selected[1])
            if N_WJets_total == 0:
                WJets_ratio =0
            else:
                WJets_ratio = float(N_WJets_selected) / N_WJets_total
            TT_ratio = float(N_TTbar_selected) / N_TTbar_total
            ST_ratio = float(N_ST_selected) / N_ST_total
            qcd_ratio = 1
            
        
        if (TT_ratio >1):
            N_selected_total = N_TTbar_total
            A = np.array([
                [1 - qcd_fraction, -qcd_fraction, -qcd_fraction],
                [-ST_fraction, 1 - ST_fraction, -ST_fraction],
                [-W_fraction, -W_fraction, 1 - W_fraction]
              ])
            b = np.array([
                qcd_fraction * N_selected_total,
                ST_fraction * N_selected_total,
                W_fraction * N_selected_total
                ])

            # Solve the system of linear equations
            N_selected = np.linalg.solve(A, b)

            N_WJets_selected = int(N_selected[2])
            N_QCD_selected = int(N_selected[0])
            N_ST_selected = int(N_selected[1])
            if N_WJets_total == 0:
                WJets_ratio =0
            else:
                WJets_ratio = float(N_WJets_selected) / N_WJets_total
            
            qcd_ratio = float(N_QCD_selected) / N_QCD_total
            ST_ratio = float(N_ST_selected) / N_ST_total
            TT_ratio = 1
        if (WJets_ratio > 1):
            A = np.array([
                [1 - qcd_fraction, -qcd_fraction, -qcd_fraction],
                [-tt_fraction, 1 - tt_fraction, -tt_fraction],
                [-ST_fraction, -ST_fraction, 1 - ST_fraction]
              ])
            b = np.array([
                qcd_fraction * N_selected_total,
                tt_fraction * N_selected_total,
                ST_fraction * N_selected_total
                ])

            # Solve the system of linear equations
            N_selected = np.linalg.solve(A, b)

            N_QCD_selected = int(N_selected[0])
            N_TTbar_selected = int(N_selected[1])
            N_ST_selected = int(N_selected[2])
            qcd_ratio = float(N_QCD_selected) / N_QCD_total
            TT_ratio = float(N_TTbar_selected) / N_TTbar_total
            if N_ST_total == 0:
                ST_ratio =0
            else:
                ST_ratio = float(N_ST_selected) / N_ST_total
       
            WJets_ratio = 1


        
        return qcd_ratio, TT_ratio, ST_ratio, WJets_ratio
    
    elif (ST_fraction != 0) and N_selected_total == N_WJets_total:
        A = np.array([
                [1 - qcd_fraction, -qcd_fraction, -qcd_fraction],
                [-tt_fraction, 1 - tt_fraction, -tt_fraction],
                [-ST_fraction, -ST_fraction, 1 - ST_fraction]
              ])
        b = np.array([
                qcd_fraction * N_selected_total,
                tt_fraction * N_selected_total,
                ST_fraction * N_selected_total
            ])

            # Solve the system of linear equations
        N_selected = np.linalg.solve(A, b)

        N_QCD_selected = int(N_selected[0])
        N_TTbar_selected = int(N_selected[1])
        N_ST_selected = int(N_selected[2])
        qcd_ratio = float(N_QCD_selected) / N_QCD_total
        TT_ratio = float(N_TTbar_selected) / N_TTbar_total
        if N_ST_total == 0:
            ST_ratio =0
        else:
            ST_ratio = float(N_ST_selected) / N_ST_total
       
        WJets_ratio = 1

        if (qcd_ratio >1):
            N_selected_total = N_QCD_total
            A = np.array([
                [1 - tt_fraction, -tt_fraction, -tt_fraction],
                [-ST_fraction, 1 - ST_fraction, -ST_fraction],
                [-W_fraction, -W_fraction, 1 - W_fraction]
              ])
            b = np.array([
                tt_fraction * N_selected_total,
                ST_fraction * N_selected_total,
                W_fraction * N_selected_total
                ])

            # Solve the system of linear equations
            N_selected = np.linalg.solve(A, b)

            N_WJets_selected = int(N_selected[2])
            N_TTbar_selected = int(N_selected[0])
            N_ST_selected = int(N_selected[1])
            WJets_ratio = float(N_WJets_selected) / N_WJets_total
            TT_ratio = float(N_TTbar_selected) / N_TTbar_total
            if N_ST_total == 0:
                ST_ratio =0
            else:
                ST_ratio = float(N_ST_selected) / N_ST_total
            qcd_ratio = 1
            
        
        if (TT_ratio >1):
            N_selected_total = N_TTbar_total
            A = np.array([
                [1 - qcd_fraction, -qcd_fraction, -qcd_fraction],
                [-ST_fraction, 1 - ST_fraction, -ST_fraction],
                [-W_fraction, -W_fraction, 1 - W_fraction]
              ])
            b = np.array([
                qcd_fraction * N_selected_total,
                ST_fraction * N_selected_total,
                W_fraction * N_selected_total
                ])

            # Solve the system of linear equations
            N_selected = np.linalg.solve(A, b)

            N_WJets_selected = int(N_selected[2])
            N_QCD_selected = int(N_selected[0])
            N_ST_selected = int(N_selected[1])
            WJets_ratio = float(N_WJets_selected) / N_WJets_total
            qcd_ratio = float(N_QCD_selected) / N_QCD_total
            if N_ST_total == 0:
                ST_ratio =0
            else:
                ST_ratio = float(N_ST_selected) / N_ST_total
            TT_ratio = 1
        return qcd_ratio, TT_ratio, ST_ratio, WJets_ratio
    
    elif ST_fraction != 0 and N_selected_total == N_QCD_total:
        N_TTbar_selected = (1-qcd_fraction)*N_selected_total/qcd_fraction
        TT_ratio = float(N_TTbar_selected) / N_TTbar_total
        qcd_ratio = 1
        ST_ratio = 0
        WJets_ratio = 0
        print(N_selected_total)
        

        if TT_ratio > 1:
            qcd_fraction = 0.99
            tt_fraction = 0.005
            W_fraction = 0.005
            ST_fraction = 0
            N_TTbar_selected = (1-qcd_fraction)*N_selected_total/qcd_fraction
            TT_ratio = float(N_TTbar_selected) / N_TTbar_total
            qcd_ratio = 1
            ST_ratio = 0
            WJets_ratio = 0
            print(N_selected_total)

        if TT_ratio > 1 or qcd_ratio > 1:
            qcd_fraction = 0.99
            tt_fraction = 0.005
            W_fraction = 0.005
            ST_fraction = 0
            N_selected_total = N_TTbar_total
            N_QCD_selected = (1-tt_fraction)*N_selected_total/tt_fraction
            qcd_ratio = float(N_QCD_selected) / N_QCD_total
            TT_ratio = 1
            ST_ratio = 0
            WJets_ratio = 0
            print(N_selected_total)
           
                
        return qcd_ratio, TT_ratio, ST_ratio, WJets_ratio


    elif (ST_fraction == 0):
        if N_selected_total == N_WJets_total:
         # Construct the matrix and the vector
            A = np.array([
                [1 - qcd_fraction, -qcd_fraction, -qcd_fraction],
                [-tt_fraction, 1 - tt_fraction, -tt_fraction],
                [-ST_fraction, -ST_fraction, 1 - ST_fraction]
              ])
            b = np.array([
                qcd_fraction * N_selected_total,
                tt_fraction * N_selected_total,
                ST_fraction * N_selected_total
            ])

            # Solve the system of linear equations
            N_selected = np.linalg.solve(A, b)

            N_QCD_selected = int(N_selected[0])
            N_TTbar_selected = int(N_selected[1])
            N_ST_selected = int(N_selected[2])
            qcd_ratio = float(N_QCD_selected) / N_QCD_total
            TT_ratio = float(N_TTbar_selected) / N_TTbar_total
            if N_ST_total == 0:
                ST_ratio =0
            else:
                ST_ratio = float(N_ST_selected) / N_ST_total
            WJets_ratio = 1
            if (qcd_ratio >1):
                N_selected_total = N_QCD_total
                A = np.array([
                    [1 - tt_fraction, -tt_fraction, -tt_fraction],
                    [-ST_fraction, 1 - ST_fraction, -ST_fraction],
                    [-W_fraction, -W_fraction, 1 - W_fraction]
                ])
                b = np.array([
                    tt_fraction * N_selected_total,
                    ST_fraction * N_selected_total,
                    W_fraction * N_selected_total
                    ])

                # Solve the system of linear equations
                N_selected = np.linalg.solve(A, b)

                N_WJets_selected = int(N_selected[2])
                N_TTbar_selected = int(N_selected[0])
                N_ST_selected = int(N_selected[1])
                WJets_ratio = float(N_WJets_selected) / N_WJets_total
                TT_ratio = float(N_TTbar_selected) / N_TTbar_total
                if N_ST_total == 0:
                    ST_ratio =0
                else:
                    ST_ratio = float(N_ST_selected) / N_ST_total
                qcd_ratio = 1
            
        
            if (TT_ratio >1):
                N_selected_total = N_TTbar_total
                A = np.array([
                    [1 - qcd_fraction, -qcd_fraction, -qcd_fraction],
                    [-ST_fraction, 1 - ST_fraction, -ST_fraction],
                    [-W_fraction, -W_fraction, 1 - W_fraction]
                ])
                b = np.array([
                    qcd_fraction * N_selected_total,
                    ST_fraction * N_selected_total,
                    W_fraction * N_selected_total
                    ])

                # Solve the system of linear equations
                N_selected = np.linalg.solve(A, b)

                N_WJets_selected = int(N_selected[2])
                N_QCD_selected = int(N_selected[0])
                N_ST_selected = int(N_selected[1])
                WJets_ratio = float(N_WJets_selected) / N_WJets_total
                qcd_ratio = float(N_QCD_selected) / N_QCD_total
                if N_ST_total == 0:
                    ST_ratio =0
                else:
                    ST_ratio = float(N_ST_selected) / N_ST_total
                TT_ratio = 1
            return qcd_ratio, TT_ratio, ST_ratio, WJets_ratio
        
        elif N_selected_total == N_QCD_total or N_selected_total == N_ST_total :
            N_selected_total = N_QCD_total
            N_TTbar_selected = (1-qcd_fraction)*N_selected_total/qcd_fraction
            TT_ratio = float(N_TTbar_selected) / N_TTbar_total
            qcd_ratio = 1
            ST_ratio = 0
            WJets_ratio = 0
            print(N_selected_total)

            if TT_ratio > 1:
                N_selected_total = N_TTbar_total
                N_QCD_selected = (1-tt_fraction)*N_selected_total/tt_fraction
                qcd_ratio = float(N_QCD_selected) / N_QCD_total
                TT_ratio = 1
                ST_ratio = 0
                WJets_ratio = 0
                print(N_selected_total)


            if TT_ratio > 1 or qcd_ratio > 1:
                qcd_fraction = 0.99
                tt_fraction = 0.005
                W_fraction = 0.005
                ST_fraction = 0
                N_selected_total = N_QCD_total
                N_TTbar_selected = (1-qcd_fraction)*N_selected_total/qcd_fraction
                TT_ratio = float(N_TTbar_selected) / N_TTbar_total
                qcd_ratio = 1
                ST_ratio = 0
                WJets_ratio = 0
                print(N_selected_total)


            if TT_ratio > 1 or qcd_ratio > 1:
                qcd_fraction = 0.99
                tt_fraction = 0.005
                W_fraction = 0.005
                ST_fraction = 0
                N_selected_total = N_TTbar_total
                N_QCD_selected = (1-tt_fraction)*N_selected_total/tt_fraction
                qcd_ratio = float(N_QCD_selected) / N_QCD_total
                TT_ratio = 1
                ST_ratio = 0
                WJets_ratio = 0
                print(N_selected_total)
                
            return qcd_ratio, TT_ratio, ST_ratio, WJets_ratio
        

def main():
    frac_file_paths = "./txt_files/background_proporitons_h_totHT_1b_2016.txt"
    nevents_file_path = "./txt_files/2016_nevents.txt"
    ht_bins = [i for i in range(1700, 9900, 200)]
    qcd_file = './h5samples/QCD_Sample_2016_BESTinputs_train_1.h5'
    tt_file = './h5samples/Top_Sample_2016_BESTinputs_train_1.h5'
    st_file = './h5samples/ST_Sample_2016_BESTinputs_train_1.h5'
    wjets_file = './h5samples/WJets_Sample_2016_BESTinputs_train_1.h5'
    output_file_path = './h5samples/bg_2016.h5'
    ratio_path='ratios_2016.txt'
    
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
if __name__ == "__main__":
    main()       


    
    

        