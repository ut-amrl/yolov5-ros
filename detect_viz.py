# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
import sys
import os
from unicodedata import name
sys.path.insert(1, os.path.abspath('../amrl_msgs/src'))
from amrl_msgs.msg import *
from utils.augmentations import letterbox
import numpy as np
from sensor_msgs.msg import Image
import cv2

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh, xywh2xyxy)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

opt = None
pub1 = None
pub2 = None

g_device = None
g_imgsz  = None
g_model  = None

def parse_labels_from_file(names):
    file = open(names, "r")
    if file.closed:
        print("failed to open file: ", names)
        exit(1)
    lines = file.readlines()
    label_id = 0
    labels_dict = dict()
    for label in lines:
        labels_dict[label_id] = label
        label_id = label_id + 1
    return labels_dict

def parse_labels_from_list(labels):
    labels_dict = dict()
    for id, label in enumerate(labels):
        labels_dict[id] = label
    return labels_dict

# assume only 1 iamge
def parse_raw(img, prediction, conf_thres=0.25):
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            return None
        boxes = xywh2xyxy(x[:, :4])
    boxes = torch.cat((boxes, x[:, 4:]), 1)

    img_viz = img.copy()
    annotator = Annotator(img_viz, line_width=1)
    for box in boxes:
        # TODO fix me
        cls = 0
        for cls_conf in box[5:]:
            if (abs(cls_conf-torch.max(box[5:]))<0.01):
                break
            cls += 1
        conf = box[4].item()
        label=f'{conf:.2f}'
        annotator.box_label([box[0], box[1], box[2], box[3]], label, color=colors(cls, True))
    return annotator.result()

@torch.no_grad()
def run(im0,
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    # Load model
    stride, names, pt = g_model.stride, g_model.names, g_model.pt

    # Dataloader
    im = letterbox(im0, g_imgsz, stride, pt)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    # Run inference
    dt, seen = [0.0, 0.0, 0.0], 0

    t1 = time_sync()
    im = torch.from_numpy(im).to(g_device)
    im = im.half() if g_model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dimclassifier
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = g_model(im, augment=augment, visualize=False)
    t3 = time_sync()
    dt[1] += t3 - t2

    # Parse Raw Output
    im_viz = parse_raw(im0, pred, conf_thres)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    det = pred[0]
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    bboxes = []
    if len(det):
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
            bboxes.append((names[c], conf, xyxy))
    im0 = annotator.result()
    return bboxes, im0, im_viz

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def viz():
    check_requirements(exclude=('tensorboard', 'thop'))
    # dir = "/robodata/taijing/ObjectSLAMDataset/yolo_viz/debug/"
    dir = "images/"
    in_dir        = dir+"inputs/"
    im0_out_dir   = dir+"outputs/nms/"
    imviz_out_dir = dir+"outputs/raw/"

    for img_name in os.listdir(in_dir):
        in_img_path    =in_dir       +img_name
        out_nms_path   =im0_out_dir  +img_name
        out_raw_path   =imviz_out_dir+img_name
        img = cv2.imread(in_img_path, cv2.IMREAD_COLOR)
        bboxes, im0, im_viz = run(img, **vars(opt))
        cv2.imwrite(out_nms_path, im0)
        if im_viz is not None:
            cv2.imwrite(out_raw_path, im_viz)
        else:
            print("did not find any candidate boxes: " + img_name)
    print("finishing...")

def prepare(opt):
    global g_device
    global g_imgsz
    global g_model

    weights = vars(opt)["weights"]
    imgsz   = vars(opt)["imgsz"]
    data    = vars(opt)["data"]
    device  = vars(opt)["device"]
    half    = vars(opt)["half"]
    dnn     = vars(opt)["dnn"]
    g_device = select_device(device)
    g_model = DetectMultiBackend(weights, device=g_device, dnn=dnn, data=data, fp16=half)
    bs = 1
    g_model.warmup(imgsz=(1 if g_model.pt else bs, 3, *imgsz))  
    g_imgsz = check_img_size(imgsz, s=g_model.stride)

if __name__ == "__main__":
    opt = parse_opt()
    prepare(opt)
    viz()