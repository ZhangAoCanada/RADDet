import sys
sys.path.insert(1, "../")
import numpy as np
import matplotlib.pyplot as plt

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

def splitAndSave(sequences, config_data):
    """ Split sequences into trainset and testset """
    np.random.shuffle(sequences)
    num_train = int(len(sequences) * config_data["trainset_portion"])
    sequences_train = sequences[:num_train]
    sequences_test = sequences[num_train:]
    writeSequences(sequences_train, "../sequences_train.txt")
    writeSequences(sequences_test, "../sequences_test.txt")
    return sequences_train, sequences_test

def writeSequences(sequences, txt_file):
    """ Write sequences into file PROJECT_ROOT/sequences_xxxx.txt """
    with open(txt_file, "w") as t:
        for i in sequences:
            t.write(str(i) + "\n")

def analyse(config_data, config_radar, sequences):
    """ Analyse the dataset statistics """
    num_instances_per_class = []
    range_analyse = []
    for i in range(len(config_data["all_classes"])):
        range_analyse.append([])
    for i in tqdm(range(len(sequences))):
        num_class_per_frame = np.zeros(len(config_data["all_classes"]))
        frame_id = sequences[i]
        gt = loader.readRadarInstances(config_data["gt_dir"], frame_id, \
                                    config_data["gt_name_format"])
        for j in range(len(gt["classes"])):
            instance_class = gt["classes"][j]
            instance_box = gt["boxes"][j]
            class_id = config_data["all_classes"].index(instance_class)
            num_class_per_frame[class_id] += 1
            range_analyse[class_id].append(int(instance_box[0]))
        num_instances_per_class.append(num_class_per_frame)
    num_instances_per_class = np.array(num_instances_per_class)
    return num_instances_per_class, range_analyse

def main():
    config = loader.readConfig(config_file_name = "../config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_process = config["PROCESS"]
    sequences = readSequences("../sequences.txt")
    sequences_train, sequences_test = splitAndSave(sequences, config_data)

    num_objs_per_class, range_analyse = analyse(config_data, config_radar, sequences_train)
    print("trainset distribution")
    print(np.sum(num_objs_per_class, axis=0))

    num_objs_per_class, range_analyse = analyse(config_data, config_radar, sequences_test)
    print("testset distribution")
    print(np.sum(num_objs_per_class, axis=0))

if __name__ == "__main__":
    main()


