import sys
sys.path.insert(1, "../../")
import numpy as np
import matplotlib.pyplot as plt

import util.loader as loader
import util.helper as helper
import util.drawer as drawer

from tqdm import tqdm

def readSequences(filename):
    """ Read sequences from PROJECT_ROOT/sequences.txt """
    sequences = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            sequence_number = int(line.split()[0])
            sequences.append(sequence_number)
    if len(sequences) == 0:
        raise ValueError("Cannot read sequences.txt. Please check if the file is organized properly.")
    return sequences

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
    config = loader.readConfig(config_file_name = "../../config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_process = config["PROCESS"]
    sequences = readSequences("../../sequences.txt")

    RAD_sample = loader.readRAD(config_data["input_dir"], sequences[0], \
                                config_data["input_name_format"])
    num_objs_per_class, range_analyse = analyse(config_data, config_radar, sequences)

    colors = loader.randomColors(config_data["all_classes"])
    ### NOTE: plot total number of objects of each class ###
    fig, axes = drawer.prepareFigure(2, figsize=(18, 8))
    drawer.barChart(np.sum(num_objs_per_class, axis=0), \
            np.arange(len(config_data["all_classes"])), \
            ax=axes[0], width=0.6, \
            title="Total number of objects on each class", \
            xlabel="number of objects", \
            ylabel="class name", color=colors, \
            ytricks=config_data["all_classes"], vertical=False, show_numbers=True)

    ### NOTE: plot total number of objects of each class ###
    drawer.barChart(np.amax(num_objs_per_class, axis=0), \
            np.arange(len(config_data["all_classes"])), \
            ax=axes[1], width=0.6, \
            title="Maximum number of objects on each class per frame", \
            xlabel="number of objects", \
            ylabel="class name", color=colors, \
            ytricks=config_data["all_classes"], vertical=False, show_numbers=True)
    fig.tight_layout(pad=5.0)
    drawer.saveFigure("../../images/dataset_statistics/", "number_objects.png")

    ### NOTE: plot range distribution of each class's objects ###
    fig, axes = drawer.prepareFigure(6, figsize=(16, 10))
    st = fig.suptitle("Range Distribution of the Objects on Each Class", \
                        fontsize="x-large")
    for i in range(len(config_data["all_classes"])):
        current_class = config_data["all_classes"][i]
        range_distribution = range_analyse[i] 
        drawer.histChart(range_distribution, 30, (0, RAD_sample.shape[0]), axes[i], \
                        title="Range distribution of " + current_class, \
                        xlabel="range bins (m)", \
                        ylabel="number of objects", \
                        color=colors[i], xticks=[0, 51, 102, 154, 205, 255], \
                        xticklabels=[0, 10, 20, 30, 40, 50])
    fig.tight_layout(pad=3.0)
    drawer.saveFigure("../../images/dataset_statistics/", "range_distribution.png")

if __name__ == "__main__":
    main()
