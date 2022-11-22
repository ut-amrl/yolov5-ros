import os
import argparse
from collections import defaultdict

import rosbag
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--bagfile', 
                    default="/robodata/taijing/uncertainty-aware-perception/yolo/yolo_1668019589.bag",
                    help="")

def read_bag_timestamps(filepath, topics):
    timestamps_dict = defaultdict(list)
    for topic, msg, t in rosbag.Bag(filepath, "r").read_messages():
        if topic not in topics:
            continue
        timestamps_dict[topic].append(t)
    return timestamps_dict

def read_pose_timestamps(filepath):
    timestamps_dict = defaultdict(list)
    timestamps_dict["pose"] = pd.read_csv(filepath, header=None).values[:,0]
    return timestamps_dict

def read_timestamps(topics, bagfile=None, filepath=None):
    timestamps_dict = {}
    if filepath:
        timestamps_dict.update(read_pose_timestamps(filepath))
    if bagfile:
        timestamps_dict.update(read_bag_timestamps(bagfile, topics))
    return timestamps_dict

def is_timestamps_ascending(timestamps_dict):
    for topic, timestamps in timestamps_dict.items():
        for i in range(1, len(timestamps)):
            if timestamps[i-1] > timestamps[i]:
                print("[Warning] Found timestamps not in ascending order! prev: ", timestamps[i-1], "; curr: ", timestamps[i])

def check(topics, bagfile=None, filepath=None):
    timestamps_dict = read_timestamps(topics, bagfile, filepath)
    is_timestamps_ascending(timestamps_dict)

if __name__ == '__main__':
    args = parser.parse_args()
    # read_bag_timestamps(filepath, ["/yolov5/bboxes", "/yolov5/im0"])
    check(["/yolov5/bboxes", "/yolov5/im0"], bagfile=args.bagfile)
