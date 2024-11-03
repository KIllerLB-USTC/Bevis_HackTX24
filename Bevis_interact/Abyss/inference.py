# -- coding: utf-8 --
import sys

from Abyss.interaction.interaction import Interactor
from hand_key.models.resnet import resnet50
from hand_track.tracker import Tracker
from hand_detection.utils.datasets import LoadImages, LoadStreams
from hand_detection.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from hand_detection.utils.torch_utils import select_device, time_synchronized
import numpy as np
import argparse
import os
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, './Abyss/hand_detection')

# os.environ['CUDA_ENABLE_DEVICES'] = '0'
# torch.cuda.set_device(0)

def detect(opt):
    # Check parameters --------------------------------------------------------------------------
    out, source, yolov5_weights, view, save = opt.output, opt.source, opt.yolov5_weights, opt.view, opt.save

    webcam = source == '0' or source.endswith('.txt') or source.startswith('rtsp') or source.startswith('http')

    pose_thres = opt.pose_thres

    res50_img_size, res50_weight = opt.res50_img_size, opt.res50_weight

    vid_path, vid_writer = None, None

    # Enable half-precision floating point if using CUDA
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Check output folder
    if not os.path.exists(out):
        os.makedirs(out)

    # Load YOLOv5 --------------------------------------------------------------------------
    model_yolo5 = torch.load(yolov5_weights, map_location=device)['model']
    model_yolo5.float().to(device).eval()
    stride = int(model_yolo5.stride.max())  # model stride
    yolov5_img_size = check_img_size(opt.yolov5_img_size, s=stride)  # check img_size
    if half:
        model_yolo5.half()  # to FP16
    names = model_yolo5.module.names if hasattr(model_yolo5, 'module') else model_yolo5.names  # Get class names
    print('load model : {}'.format(yolov5_weights))

    # Load ResNet50 --------------------------------------------------------------------------
    model_res50 = resnet50(num_classes=42, img_size=res50_img_size[0])
    model_res50.to(device).eval()
    if half:
        model_res50.half()

    chkpt = torch.load(res50_weight, map_location=device)
    model_res50.load_state_dict(chkpt)
    print('load model : {}'.format(res50_weight))

    # Initialize tracker --------------------------------------------------------------------------
    tracker = Tracker(opt.pose_cfg, pose_thres=pose_thres)

    # Initialize interaction module --------------------------------------------------------------------------
    interactor = Interactor()

    # Dataloader --------------------------------------------------------------------------
    if webcam:
        view = check_imshow()
        cudnn.benchmark = True  # Accelerate inference with constant size images in video
        dataset = LoadStreams(source, img_size=yolov5_img_size, stride=stride)
    else:
        view = True
        dataset = LoadImages(source, img_size=yolov5_img_size, stride=stride)

    # Start inference --------------------------------------------------------------------------------------------
    t0 = time.time()
    # Warm-up models
    img = torch.zeros((1, 3, yolov5_img_size, yolov5_img_size), device=device)
    _ = model_yolo5(img.half() if half else img) if device.type != 'cpu' else None
    img = torch.zeros((1, 3, res50_img_size[0], res50_img_size[0]), device=device)
    _ = model_res50(img.half() if half else img) if device.type != 'cpu' else None

    # Process each image/frame (path to file, YOLO size (3, h, w), original image (h, w, 3), none)
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        # YOLO part image preprocessing --------------------------------------------------------------------------
        img = torch.from_numpy(img).to(device)  # Convert image to tensor and assign device
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # Scale 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # Add one dimension

        # Inference
        t1 = time_synchronized()
        pred = model_yolo5(img, augment=opt.augment)[0]  # Get prediction results list

        # Non-maximum suppression, max detection is 2, min box side length: fist length * 0.9
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=[0])

        # Extract data from pred list, only one loop if it's a video, contains tensor[2,6] 2 hands, 6 pieces of info
        for b, det_box in enumerate(pred):
            if webcam:  # batch_size >= 1
                p, s, im0 = path[b], '%g: ' % b, im0s[b].copy()
            else:
                p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]
            save_path = str(Path(out) / Path(p).name)

            if det_box is not None and len(det_box):  # If a target is detected, track the states of two hands and display tracking effect
                # Scale prediction boxes back to original coordinates from yolov5_img_size
                det_box[:, :4] = scale_coords(img.shape[2:], det_box[:, :4], im0.shape).round()

                # Hand detection count
                s += '%g %ss ' % (len(det_box), names[0])

                # Predict keypoints for each hand and return original coordinates
                keypoint_list = []
                for one_hand_box in det_box.data.cpu().numpy():
                    # ResNet50 image preprocessing -----------------------------------------------------------------
                    cut_img = im0[int(one_hand_box[1]):int(one_hand_box[3]),
                                  int(one_hand_box[0]):int(one_hand_box[2])]  # Crop y-axis first, then x-axis

                    # cv2.imshow('test', cut_img)
                    # cv2.waitKey(0)

                    key_img = cv2.resize(cut_img, (res50_img_size[1], res50_img_size[0]),
                                         interpolation=cv2.INTER_CUBIC)  # Resize to ResNet50 input size
                    key_img = (key_img.astype(np.float32) - 128.) / 256.
                    key_img = torch.from_numpy(key_img.transpose(2, 0, 1)).unsqueeze_(0)
                    if torch.cuda.is_available():
                        key_img = key_img.cuda()
                    key_img = key_img.half() if half else key_img.float()
                    # Model inference
                    key_output = model_res50(key_img)
                    key_output = key_output.cpu().detach().numpy()
                    key_output = np.squeeze(key_output)  # Prediction range [0,1]
                    hand_ = []
                    for i in range(int(key_output.shape[0] / 2)):
                        x = (key_output[i * 2 + 0] * float(cut_img.shape[1])) + int(one_hand_box[0])
                        y = (key_output[i * 2 + 1] * float(cut_img.shape[0])) + int(one_hand_box[1])
                        hand_.append([x, y])
                    keypoint_list.append(hand_)

                # Process result: track & print ------------------------------------------------------------------------
                tracker.update(det_box, keypoint_list)

            else:
                tracker.update_nodata([0, 1])  # Ignore if tracking is lost due to fast movement

            unprocessed_img = im0.copy()

            tracker.plot(im0)

            # Enter other functions here
            interactor.interact(im0, unprocessed_img, tracker.get_order())

            # Print time (yolov5 + NMS + keypoint + track + draw + interact + ...)
            t2 = time_synchronized()
            print('%s (%.3fs)' % (s, t2 - t1))  # Time

            # Stream results  Display in UI if enabled
            if view is None:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save:
                if dataset.mode == 'images':
                    print('saving img!')
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save:
        print('Results saved to %s' % os.getcwd() + os.sep + out)

    print('Done. (%.3fs)' % (time.time() - t0))

def parse_argument():
    parser = argparse.ArgumentParser()
    # file/folder, 0-webcam. Image format not supported
    parser.add_argument('--source', type=str, default='inference/input/piano.mp4', help='source')
    # Output folder
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    # Output video format
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # Display results
    parser.add_argument('--view', default=True, help='display results')
    # Save video
    parser.add_argument("--save", type=str, default=False, help='save results')
    # Use GPU + half-precision
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # YOLOv5 model path
    parser.add_argument('--yolov5_weights', type=str, default='inference/weights/hand_weight/best_YOLOv5l.pt')
    # YOLOv5 input size
    parser.add_argument('--yolov5_img_size', type=int, default=640, help='inference size (pixels)')
    # YOLOv5 augmented inference
    parser.add_argument('--augment', action='store_true', default=False, help='augmented inference')
    # NMS confidence threshold
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    # NMS IoU threshold
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    # Tracker pose angle threshold
    parser.add_argument("--pose_thres", type=float, default=0.4, help='pose angle threshold')
    # Tracker gesture dictionary config file
    parser.add_argument("--pose_cfg", type=str, default='inference/weights/cfg_pose.json', help='pose_cfg')
    # ResNet50 model path
    parser.add_argument('--res50_weight', type=str, default='inference/weights/pose_weight/resnet50_2021-418.pth',
                        help='res50_weight')
    # ResNet50 input size
    parser.add_argument('--res50_img_size', type=tuple, default=(256, 256), help='res50_img_size')

    opt = parser.parse_args()
    print(opt)
    return opt

if __name__ == '__main__':
    with torch.no_grad():
        detect(parse_argument())
