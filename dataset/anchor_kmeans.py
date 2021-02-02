import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tqdm import tqdm

import sys
sys.path.insert(1, '../')
import util.loader as loader
import util.helper as helper
import util.drawer as drawer

def loadBoxSizes(config_data, sequences):
    """ load all boxes from ground truth annotations """
    ### TODO: do we need to separate boxes according to the classes ? ###
    all_boxes = []
    for idx in tqdm(range(len(sequences))):
        frame_id = sequences[idx]
        gt_instances = loader.readRadarInstances(config_data["gt_dir"], \
                                frame_id, config_data["gt_name_format"])
        ### NOTE: boxes: [x_center, y_center, z_center, w, h, d] ###
        for box_i in gt_instances["boxes"]:
            ### this is for [x_min, x_max, y_min, y_max, z_min, z_max] ###
            # box_whd = helper.boxLocationsToWHD(box_i)
            # all_boxes.append(box_whd[3:])
            all_boxes.append(box_i[3:])

    print("total number of gt boxes : ", len(all_boxes))
    if len(all_boxes) == 0:
        raise ValueError("No box is loaded, please check if the ground \
                        truth directory is right or not.")
    else:
        return np.array(all_boxes)

def getBoxesFromGt():
    """ Read config, load ground truth, and find box sizes """
    config = loader.readConfig(config_file_name = "../config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_process = config["PROCESS"]
    assert isinstance(config_process["sequences"], list)
    sequences = loader.getSequenceNumbers(config_data["gt_dir"], \
                                        config_data["gt_name_format"])
    gt_boxes = loadBoxSizes(config_data, sequences)
    return gt_boxes

def boxesKmeans(all_boxes, number_of_centers):
    """ Find k means centers for all the boxes """
    kmeans = KMeans(n_clusters=number_of_centers, random_state=0).fit(all_boxes)
    return kmeans.cluster_centers_, kmeans.inertia_

def getCenters(times):
    """ Generate a list with element standing for number of kmeans centers """
    """ It always starts with 0, as for plotting kmeans for statistic """
    assert isinstance(times, int)
    centers = []
    for i in range(1, times):
        centers.append(i)
    return centers

def getOptimalAnchorsWithPlot(gt_boxes, centers, stopping_err_portion=0.90):
    """ Calculate optimal anchor boxes, plot kmean errors for visualization """
    centers_explored = []
    centers_errs = []
    previous_err = None
    previous_center = None
    for i in centers:
        kmeans_centers, kmeans_errors = boxesKmeans(gt_boxes, i)
        if previous_err is None or kmeans_errors < stopping_err_portion*previous_err:
            previous_err = kmeans_errors
            previous_center = kmeans_centers
            centers_explored.append(i)
            centers_errs.append(kmeans_errors)
        else:
            centers_explored.append(i)
            centers_errs.append(kmeans_errors)
            break
    ### NOTE: draw kmeans errors bar chart for visualization ###
    fig, axes = drawer.prepareFigure(1, figsize=(7, 6))
    drawer.barChart(centers_explored, centers_errs, axes[0], \
                    title="K means erros", xlabel="Number of kmeans centers", \
                    ylabel="Values of kmeans errors", color="blue", xtricks=centers_explored)
    drawer.saveFigure("../images", "anchor_kmeans.png")
    return previous_center

def saveAnchorsAsTxt(anchors):
    """ Save anchor boxes to txt file """
    with open("../anchors.txt", "w") as f:
        for i in range(anchors.shape[0]):
            for j in range(anchors.shape[1]):
                f.write(str(int(np.round(anchors[i,j]))))
                f.write(" ")
            f.write("\n")

def main():
    """ Find and store anchors """
    centers = getCenters(100)
    print("------------- gt loading ---------------")
    gt_boxes = getBoxesFromGt()
    print("------------- Kmeans finding --------------")
    anchors = getOptimalAnchorsWithPlot(gt_boxes, centers)
    print("Number of anchors: ", anchors.shape[0])
    saveAnchorsAsTxt(anchors)

if __name__ == "__main__":
    main()
