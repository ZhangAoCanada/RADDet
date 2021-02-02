import numpy
import sys
sys.path.insert(1, '../')
import util.loader as loader
import util.helper as helper
import util.drawer as drawer
from tqdm import tqdm

def getSequences(config_data):
    """ Read config, load ground truth, and find box sizes """
    sequences_input = loader.getSequenceNumbers(config_data["gt_dir"], \
                                        config_data["gt_name_format"])
    sequences_gt = loader.getSequenceNumbers(config_data["gt_dir"], \
                                        config_data["gt_name_format"])
    return sequences_input, sequences_gt

def writeSequences(sequences):
    """ Write sequences into file PROJECT_ROOT/sequences.txt """
    txt_file = "../sequences.txt"
    with open(txt_file, "w") as t:
        for i in sequences:
            t.write(str(i) + "\n")

def main():
    """ Read sequences and write sequences into txt file """
    config = loader.readConfig(config_file_name = "../config.json")
    config_data = config["DATA"]
    sequences_input, sequences_gt = getSequences(config_data)
    print("------------- first-hand loading results ---------------")
    print("Input directory has Frames: ", len(sequences_input))
    print("Gt directory has Frames: ", len(sequences_gt))
    sequences_common = []
    print("------------- start processing sequences ---------------")
    for idx in tqdm(range(len(sequences_input))):
        i = sequences_input[idx]
        if i in sequences_gt:
            RAD_complex = loader.readRAD(config_data["input_dir"], i, \
                                        config_data["input_name_format"])
            gt_instances = loader.readRadarInstances(config_data["gt_dir"], i, \
                                                    config_data["gt_name_format"])
            if RAD_complex is not None and gt_instances is not None:
                sequences_common.append(i)

    print("-------------- sequence numbers loading results ----------------")
    print("There are totally ", len(sequences_common), " Frames in the dataset")
    print("----------------------------------------------------------------")

    if len(sequences_common) > 0:
        writeSequences(sequences_common)
    else:
        raise ValueError("No sequence is being extracted, please check if the input directory (or format) or ground truth directory (or format) is right.")

if __name__ == "__main__":
    main()


