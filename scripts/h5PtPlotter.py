import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents opening displays
import matplotlib.pyplot as plt
import h5py
import argparse, os, sys

# User definitions
bins_list = [i * 50 for i in range(20, 200)]

# Global variables
years = ["combine"]
sampleFileTypes = ["bg", "allDecays"]
sampleTypes = ["bg", "allDecays"]
listOfFileTypes = ["_train_flattened.h5"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples', dest='samples', default="all")
    parser.add_argument('-hd', '--h5Dir', dest='h5Dir', default="../formatConverter/h5samples/")
    parser.add_argument('-y', '--years', dest='years', default="all")
    parser.add_argument('-pt', '--ptIndex', dest='ptIndex', type=int, default=111)
    parser.add_argument('-ft', '--fileTypes', dest='fileTypes', default="all")
    args = parser.parse_args()

    if args.samples != "all": sampleTypes = args.samples.split(',')
    if args.fileTypes != "all": listOfFileTypes = args.fileTypes.split(',')
    if args.years != "all": years = args.years.split(',')

    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        sys.exit()

    for year in years:
        for fileType in listOfFileTypes:
            plotDir = "plots/prettyHT/{}/".format(fileType[1:-3])
            if not os.path.exists(plotDir):
                os.makedirs(plotDir)

            myPtArrays = []
            labels = []  # CHANGED: Added label list for legend
            colors = []  # CHANGED: Added color list for legend
            
            for sampleType in sampleFileTypes:
                inputPath = os.path.join(args.h5Dir, "{}_Sample_all_mass_{}_BESTinputs{}".format(sampleType, year, fileType))
                try:
                    inputFile = h5py.File(inputPath, "r")
                except (OSError, IOError):
                    print("Error: Cannot open {}".format(inputPath))
                    continue

                if sampleType == "bg":
                    data = np.array(inputFile["BES_vars"])
                    sample_labels = ['QCD','TT','WJets','ST']  # CHANGED: Defined labels for bg
                    sample_colors = ['blue', 'green', 'red', 'purple']  # CHANGED: Defined colors for bg
                    
                    for i in [3, 2, 1, 0]:
                        myPtArrays.append(data[data[..., 112] == i][..., args.ptIndex])
                        labels.append(sample_labels[i])  # CHANGED: Append correct label
                        colors.append(sample_colors[i])  # CHANGED: Append correct color
                else:
                    myPtArrays.append(np.array(inputFile["BES_vars"][..., args.ptIndex]))
                    labels.append("allDecays")  # CHANGED: Added allDecays label
                    colors.append("black")  # CHANGED: Assigned color for allDecays

            plt.figure()
            plt.hist(myPtArrays[:-1], bins=bins_list, histtype='stepfilled', label=labels[:-1], stacked=True, color=colors[:-1], alpha=0.7, log=True)
            plt.hist(myPtArrays[-1], bins=bins_list, histtype='step', label=labels[-1], color=colors[-1], log=True)  # CHANGED: Use defined labels/colors
            plt.legend(frameon=True, loc='upper right')  # CHANGED: Ensure legend includes all samples
            plt.xlabel('HT (GeV)')
            plt.title("HT Distribution {}".format(year))
            plt.ylim(bottom=0.1)

            savePath = os.path.join(plotDir, "HT_Distribution_{}.png".format(year))
            plt.savefig(savePath)
            plt.xlim([1400, 10000])  # CHANGED: Ensure zoomed version is correctly modified
            savePath_zoomed = os.path.join(plotDir, "HT_Distribution_{}_zoomed.png".format(year))
            plt.savefig(savePath_zoomed)    
            plt.clf()
            plt.close()

    print("Done")