# YOLOv5 ROS Documentation

To use yolov5 with ros, run:
```
python detect_ros.py --weights runs/train/exp2/weights/best.pt --img <resolution> --conf <conf>
```
By default, it subscribes to topic `/camera/rgb/image_raw` and publish bouding box messages by `/yolov5/bboxes`. If you want to see the images annotated with bounding boxes, they're published by topic `/yolov5/im0`. You can change topic it subscribes to in the main function in `detect_ros.py`.