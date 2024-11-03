# -- coding: utf-8 --
import _thread

import cv2
import numpy as np

from Abyss.interaction.audio_thread import play_sounds
from Abyss.interaction.resnet_classfication.classfication_thread import flower_classfication
from Abyss.interaction.utils import compute_distance, compute_direction
from server.text_tts_for_oneimg import text_tts_for_oneimg

action_list = {
    "8": "close",
    "2": "menu",
    "6": "forward",
    "4": "backward",
    "68": "copy",
    "62": "paste",
    "64": "start",  # Launch application
    "46": "clear",  # Refresh
}

class Interactor:
    """
    User interaction module:
    Further processes the results obtained from the Tracker.
    1. Determines whether the user has an intention to interact
    2. Enhances robustness
    3. Calls specific applications
    """

    def __init__(self, no_act_thres=20, stop_thres=40, stable_thres=40):
        # Settings
        self.no_act_thres = no_act_thres  # Tolerable number of incorrect frames
        self.stop_thres = stop_thres  # Distance to determine if movement is stationary
        self.stable_thres = stable_thres  # Time to determine if stationary action is triggered

        # Cache
        self.pose = np.array([0, 0])  # Registered response gestures (in the current version, only single gesture movements are supported, and actions will be canceled when changing gestures)
        self.pose_will = np.array([0, 0])  # Upcoming gesture, used as a cache
        self.direction_list = ["", ""]  # Direction list, including up 8, down 2, left 4, right 6
        self.track = ([], [])  # Response tracking path -> draw graph
        self.stable_time = np.array([0, 0])  # Accumulated stationary time
        self.no_active = np.array([0, 0])  # Ignore count
        self.pose_end = np.array([0, 0])  # Mark voice output
        self.direction_intent = np.array([0, 0])  # Movement direction prediction
        self.processing_flag = False  # Voice processing flag
        self.text = "Bevis Ready"
        # Multithreading
        self.share_act = []
        _thread.start_new_thread(play_sounds, (self.share_act,))  # Start sound notification thread

        ## text_tts_for_oneimg_thread

        # Application module
        self.app_dict = {
            "mouse": None,
            "img": None, 
            "left_top": None,
            "right_bottom": None,
            "result": None,
        }  # Shared information
        self.app_start = False  # todo Add additional application modules 
        _thread.start_new_thread(flower_classfication, (self.app_dict,))  # Start application module thread

    def run_text_tts_with_flag(self,img_path,img):
        try:
            self.text = text_tts_for_oneimg(img_path)
            cv2.putText(img, self.text, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
        finally:
            self.processing_flag = False

    def interact(self, im0, unprocessed_img, order):
        """
        order[[x,y,p],[x,y,p]], stored in sequence by id
        """
        hands_1_gesture = [False, False]  # Records whether both hands are in gesture "1"
        stable_frames = 20  
        click_positions = [(0, 0), (0, 0)]  # Stores click positions of both hands
        
        for i, o in enumerate(order):
            # --------------------------------------- Termination gesture
            if o[2] == 10 or o[2] == 8 or o[2] == 5:  # Clear memory and send instruction
                self.action(i, im0)
                self.__clear_cache(i)

            # elif o[2] == 5:  # Clear memory without sending instruction
            #     self.__clear_cache(i)

            # --------------------------------------- No gesture or not the registered response gesture or no detection target
            elif o[2] != 1 and o[2] != 2 or o[2] != self.pose_will[i] or o[0] == 0:
                if self.no_active[i] == self.no_act_thres:
                    self.__clear_cache(i)
                else:
                    self.no_active[i] += 1
                    self.pose_will[i] = o[2]  # Only respond after passing detection threshold

            # --------------------------------------- Response gesture
            elif o[2] == 1 or o[2] == 2:
                self.no_active[i] = 0
                self.pose[i] = o[2]  # Update status

                if o[2] == 1 and not self.pose_end[i]:  # Real-time detection for gesture "1"
                    hands_1_gesture[i] = True
                    click_positions[i] = (o[0], o[1])
                    if hands_1_gesture[0] and hands_1_gesture[1]:
                        if self.stable_time[i] >= stable_frames:
                           detect_flag = True
                           x1, y1 = click_positions[0]
                           x2, y2 = click_positions[1]
                           cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                           cv2.putText(im0, "select", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                           x1,x2 = sorted([x1,x2])
                           y1,y2 = sorted([y1,y2])
                           crop_img = unprocessed_img[y1:y2, x1:x2]
                           window_name = "Cropped Image"
                           cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                           cv2.moveWindow(window_name, 400, 300) 
                           cv2.imshow(window_name, crop_img)
                           cv2.imwrite("crop.jpg", crop_img)
                           if not self.processing_flag:
                                self.processing_flag = True
                                _thread.start_new_thread(self.run_text_tts_with_flag, ("crop.jpg",crop_img,))
                           cv2.waitKey(20)
                           self.share_act.append("select")

                    if not self.direction_list[i]:  # No movement    
                        if self.stable_time[i] == 1:
                            self.share_act.append("click")  # Click
                        elif self.stable_time[i] == self.stable_thres - 1:
                            self.share_act.append("press")  # Long press
                    elif len(self.track[i]) != 1:
                        self.share_act.append("drag")  # Drag
                        self.pose_end[i] = 1

                if not len(self.track[i]):  # First record, no coordinates
                    self.track[i].append(o.tolist())

                # Judged as stationary, do not update direction or track
                if compute_distance(o[0], o[1],
                                    self.track[i][len(self.track[i]) - 1][0],
                                    self.track[i][len(self.track[i]) - 1][1]) < self.stop_thres ** 2:

                    if self.stable_time[i] < self.stable_thres and not len(self.direction_list[i]):
                        self.stable_time[i] += 1
                # Judged as movement
                else:
                    direction = compute_direction(o[0], o[1], self.track[i][len(self.track[i]) - 1][0],
                                                  self.track[i][len(self.track[i]) - 1][1])
                    self.track[i].append(o.tolist())  # Record coordinates of response track

                    if not self.direction_list[i]:  # First movement, no direction
                        self.direction_list[i] = str(direction)

                    if self.direction_intent[i] == direction:
                        if not self.direction_list[i].endswith(str(direction)):
                            self.direction_list[i] = self.direction_list[i] + str(direction)  # Update if new direction
                    else:
                        self.direction_intent[i] = direction

                    self.stable_time[i] = 0  # Do not update stationary record when moving

            else:
                print("wrong in interaction")

            # --------------------------------------- Draw graph
            if len(self.track[i]):  # Check if there is a record
                last = None  # Create variable to store the previous record in the loop
                for t, p in enumerate(self.track[i]):  # Iterate through each record
                    if last is None:  # First record
                        last = (p[0], p[1])
                    else:
                        cv2.line(im0, last, (p[0], p[1]), (240, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                        last = (p[0], p[1])

                fill_cnt = (self.stable_time[i] / self.stable_thres) * 360
                cv2.circle(im0, (o[0], o[1]), 5, (0, 150, 255), -1)
                if 0 < fill_cnt < 360:
                    cv2.ellipse(im0, (o[0], o[1]), (self.stop_thres, self.stop_thres), 0, 0, fill_cnt, (255, 255, 0), 2)
                else:
                    cv2.ellipse(im0, (o[0], o[1]), (self.stop_thres, self.stop_thres), 0, 0, fill_cnt, (0, 150, 255), 4)

                cv2.line(im0, (o[0], o[1]), last, (240, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # todo Draw box done
        if self.app_dict["left_top"] is not None and self.app_dict["right_bottom"] is not None:
            cv2.rectangle(im0, (self.app_dict["left_top"][0], self.app_dict["left_top"][1]),
                          (self.app_dict["right_bottom"][0], self.app_dict["right_bottom"][1]),
                          (0, 255, 127), 2)
            if self.app_dict["result"] is not None:
                cv2.putText(im0, self.app_dict["result"]["class"] + self.app_dict["result"]["score"],
                            (self.app_dict["left_top"][0], self.app_dict["left_top"][1]),
                            cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 255], 2)

    def __clear_cache(self, i):
        self.track[i].clear()  # Clear tracking record
        self.direction_list[i] = ""  # Clear direction record
        self.no_active[i] = 0
        self.pose[i] = 0
        self.pose_will[i] = 0
        self.stable_time[i] = 0
        self.pose_end[i] = 0

    def action(self, i, img=None):
        if self.pose[i] == 2 and self.direction_list[i] in action_list.keys():  # Need to register command
            self.share_act.append(action_list[self.direction_list[i]])

            if self.direction_list[i] == "64":  # todo Launch application done
                self.app_start = True
            elif self.direction_list[i] == "8":  # todo Close application done
                self.app_start = False
            elif self.direction_list[i] == "46":  # todo Clear to create new classification task done
                self.app_dict["result"] = None
                self.app_dict["left_top"] = self.app_dict["right_bottom"] = None

        if self.pose[i] == 1 and self.app_start and self.app_dict["result"] is None:  # todo Start classification done
            self.app_dict["img"] = img
