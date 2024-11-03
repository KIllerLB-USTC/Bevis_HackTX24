# -- coding: utf-8 --
import _thread
import json
import cv2
import numpy as np
import torch

from Abyss.hand_track.angle_util import pose_to_angles, piano_judge
from Abyss.hand_track.draw_util import draw
#from Abyss.interaction.audio_thread import play_piano


class Tracker:
    """
    Detection and tracking module.
    Performs preliminary processing of the deep learning model results.
    1. Enhances robustness
    2. Recognizes gestures
    3. IOU tracking
    """

    def __init__(self, pose_cfg, pose_thres=0.2, no_data_limit=5):
        # Settings
        with open(pose_cfg, 'r') as f:
            self.pose_list = json.load(f)  # Gesture rules list[dict{name:int, angle:list[float*5]}*n gestures]
        self.pose_thres = pose_thres
        self.no_data_limit = no_data_limit  # Tolerance for the number of frames with no detection

        # Cache
        self.last_box = np.array([[0, 0, 0, 0],
                                  [0, 0, 0, 0]])  # Box position of the previous frame
        self.no_data_now = 0  # Current count of detection loss
        self.active_click = np.array([[0, 0, 0],
                                      [0, 0, 0]])  # Click response position + static gesture
        self.plot_cache = [[None, None],
                           [None, None]]

        # Multithreading
        # self.piano_data = []
        # _thread.start_new_thread(play_piano, (self.piano_data,))
        self.circle_list = []
    

    def update(self, det, key_point_list):
        # Return format
        for n, (*xyxy, conf, cls) in enumerate(det):  # Iterate over possible two hands, six parameters
            x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            if isinstance(conf, torch.Tensor):
                conf = conf.item()
            iou: np.ndarray = self.__compute_iou(x1, y1, x2, y2)
            # Get a 2D array, corresponding to IOU of hand gestures for id 0/1
            track_id, iou_val = iou.argmax(), iou.max()  # Get the hand with the maximum IOU for initial tracking
            pose: int = self.__compute_pose(key_point_list[n])

            # Piano
            piano = False
            if piano:
                self.circle_list = piano_judge(key_point_list[n], self.piano_data)

            # Update internal tracking
            if iou_val == 0:  # The current hand doesn't match either of the previously recorded hands
                if self.last_box[track_id].max() != 0:  # The currently tracked hand has a record
                    if self.last_box[1 - track_id].max() != 0:  # The other hand in the record also has a record
                        self.update_nodata([0, 1])  # ① Moving too fast: reset everything
                        return
                    else:  # The other hand in the record doesn't have a record
                        track_id = 1 - track_id  # ② A new hand appeared in the scene, correct the id

            self.no_data_now = 0  # Reset detection loss count

            # ③ Successfully tracked the moving hand
            self.last_box[track_id] = np.array([x1, y1, x2, y2])
            self.active_click[track_id][0] = key_point_list[n][8][0]
            self.active_click[track_id][1] = key_point_list[n][8][1]
            self.active_click[track_id][2] = pose
            self.plot_cache[track_id][0] = np.array([x1, y1, x2, y2, conf, track_id, iou_val, pose])
            self.plot_cache[track_id][1] = key_point_list[n]

            if len(det) == 1:
                self.update_nodata(1 - track_id, now=True)

    def plot(self, im0):
        for track_id in range(0, 2):
            if self.plot_cache[track_id][0] is not None:
                draw(im0, self.plot_cache[track_id][0], self.plot_cache[track_id][1])
        for i, point in enumerate(self.circle_list):
            x = int(point[0])
            y = int(point[1])
            c = int(255 * (i + 1) / 6)
            cv2.circle(im0, (x, y), 25, (c, 255 - c, abs(122 - c)), 5)

    def get_order(self):
        """
        Get response position and gesture
        """
        return self.active_click

    def update_nodata(self, idx, now=False):
        """
        Clear recorded data
        """
        if now or self.no_data_now == self.no_data_limit:
            self.last_box[idx] = np.array([0, 0, 0, 0])
            self.active_click[idx] = np.array([0, 0, 0])
            if idx == 1 or idx == 0:
                self.plot_cache[idx][0] = None
                self.plot_cache[idx][1] = None
            else:
                self.plot_cache = [[None, None],
                                   [None, None]]
            self.no_data_now = 0
        else:
            self.no_data_now += 1

    def __compute_iou(self, x1, y1, x2, y2):
        """
        Calculate IOU value between the current prediction box and the recorded two prediction boxes
        """
        box1 = np.array([x1, y1, x2, y2])
        iou_list = []
        for box2 in self.last_box:
            h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
            w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
            area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
            area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
            inter = w * h
            union = area_box1 + area_box2 - inter
            iou = inter / union
            iou_list.append(iou)

        iou_list = np.array(iou_list)
        return iou_list

    def __compute_pose(self, key_point):
        """
        Read settings file and match gestures
        """
        angles = pose_to_angles(key_point)  # [0.99953, -0.91983, -0.95382, -0.98989, -0.99999]
        for pose in self.pose_list:
            max = (np.array(pose['angle']) + self.pose_thres >= angles).sum()
            min = (np.array(pose['angle']) - self.pose_thres <= angles).sum()
            if max == min == 6:
                return int(pose['name'])
        return 0


if __name__ == '__main__':
    t = Tracker()
    # t.last_time[0] = 90
    # t.last_time[1] = 90
    # t.get_order()
    # t.update_nodata([0, 1])
    # t.__compute_iou(1, 1, 5, 5)
    # t.last_box += 1
    # print(t.last_box)
    # print(t.last_time[0])
    # print(n)
    # print(type(n))
