import numpy as np
import sys
sys.path.insert(1, '../')
import util.loader as loader
import util.helper as helper
import util.drawer as drawer
from tqdm import tqdm

def readSequences(filename):
    """ Read sequences from PROJECT_ROOT/sequences_xxxx.txt """
    sequences = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            sequence_number = int(line.split()[0])
            sequences.append(sequence_number)
    if len(sequences) == 0:
        raise ValueError("Cannot read sequences.txt. Please check if the file is organized properly.")
    return sequences

def main():
    config = loader.readConfig(config_file_name = "../config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_process = config["PROCESS"]
    sequences = readSequences("../sequences.txt")
    all_means = []
    all_maxs = []
    all_mins = []

    for idx in tqdm(range(len(sequences))):
        i = sequences[idx]
        RAD_complex = loader.readRAD(config_data["input_dir"], i, \
                                    config_data["input_name_format"])
        RAD_data = helper.complexTo2Channels(RAD_complex)
        all_means.append(RAD_data.mean())
        all_maxs.append(RAD_data.max())
        all_mins.append(RAD_data.min())

    print("global mean: ", np.mean(all_means))
    print("global max: ", np.amax(all_maxs))
    print("global min: ", np.amin(all_mins))
    print("global variance: ", \
            np.maximum(np.abs(np.mean(all_means) - np.amin(all_mins)), \
            np.abs(np.mean(all_means) - np.amax(all_maxs))) \
            )


if __name__ == "__main__":
    main()



