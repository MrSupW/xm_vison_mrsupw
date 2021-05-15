#!/usr/bin/python3
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import argparse
import random
import sys
import time

import cv2
import rospy
from geometry_msgs.msg import *
from pyk4a import PyK4A, ColorResolution
from xm_msgs.msg import *
from xm_msgs.srv import *

sys.path.append("/home/xm/xm_vision/darknet")
import darknet

codeCodes = {
    'black': '0;30', 'bright gray': '0;37',
    'blue': '0;34', 'white': '1;37',
    'green': '0;32', 'bright blue': '1;34',
    'cyan': '0;36', 'bright green': '1;32',
    'red': '0;31', 'bright cyan': '1;36',
    'purple': '0;35', 'bright red': '1;31',
    '***': '0;33', 'bright purple': '1;35',
    'grey': '1;30', 'bright yellow': '1;33',
}


def colored(text, color='green'):
    return "\033[" + codeCodes[color] + "m" + text + "\033[0m"


def image_detection_original_no_image(image, network, class_names, thresh, fx, fy):
    global width, height
    darknet_image = darknet.make_image(width, height, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    for i in range(len(detections)):
        detections[i] = list(detections[i])
        detections[i][2] = list(detections[i][2])
        detections[i][2][0] = (width / 2 / fx) - (1 / fx) * (width / 2 - detections[i][2][0]) if detections[i][2][
                                                                                                     0] <= width / 2 \
            else (1 / fx) * (detections[i][2][0] - width / 2) + (width / 2 / fx)
        detections[i][2][1] = (height / 2 / fy) - (1 / fy) * (height / 2 - detections[i][2][1]) if detections[i][2][
                                                                                                       1] <= height / 2 \
            else (1 / fy) * (detections[i][2][1] - height / 2) + (height / 2 / fy)
        detections[i][2][2] /= fx
        detections[i][2][3] /= fy
    darknet.free_image(darknet_image)
    return detections


# 从所有的检测结果中筛选出来在摄像头5m范围的人、沙发和椅子
def filter_detections(detections, depth_point_cloud):
    global persons, chairs, sofa
    new_detection = []
    for detection in detections:
        if detection[0] == 'person' and sum(
                [(v / 1000) ** 2 for v in depth_point_cloud[int(detection[2][1])][int(detection[2][0])]]) ** 0.5 <= 5:
            persons.append(detection)
            new_detection.append(detection)
        elif detection[0] == 'chair' and sum(
                [(v / 1000) ** 2 for v in depth_point_cloud[int(detection[2][1])][int(detection[2][0])]]) ** 0.5 <= 5:
            chairs.append(detection)
            new_detection.append(detection)
        elif detection[0] == 'sofa' and sum(
                [(v / 1000) ** 2 for v in depth_point_cloud[int(detection[2][1])][int(detection[2][0])]]) ** 0.5 <= 5:
            sofa = detection
            new_detection.append(detection)
    return new_detection


def judge_chairs_empty(persons, chairs):
    global width
    for chair in chairs:
        empty = True
        for person in persons:
            if chair[2][0] - chair[2][2] // 2 <= person[2][0] <= chair[2][0] + chair[2][2] // 2:
                empty = False
                break
        if empty:
            return True, 0 if (chair[2][0] < width / 2) else 4
    return False, -1


def judge_chair_empty(persons, chairs):
    chair = chairs[0]
    empty = True
    for person in persons:
        if chair[2][0] - chair[2][2] // 2 <= person[2][0] <= chair[2][0] + chair[2][2] // 2:
            empty = False
            break
    if empty:
        return True, 0 if (chair[2][0] < width / 2) else 4
    return False, -1


def judge_sofa_empty(persons, sofa):
    position_slice = [(sofa[2][0] - sofa[2][2] // 2) + i * (sofa[2][2] // 3) for i in range(4)]
    flags = [True, True, True]
    for person in persons:
        for i in range(len(flags)):
            if position_slice[i] <= person[2][0] <= position_slice[i + 1]:
                flags[i] = False
    for i in range(len(flags)):
        if flags[i]:
            return True, i + 1
    return False, -1


def find_vacant_seat(msg):
    global is_debug,fx,fy
    count = 0
    while not rospy.is_shutdown():
        global persons, chairs, sofa
        persons = []
        chairs = []
        sofa = ''
        start_time = time.time()
        capture = camera.get_capture()
        detections = filter_detections(
            image_detection_original_no_image(capture.color, network, class_colors, 0.3, fx, fy),
            capture.transformed_depth_point_cloud)
        print('persons', persons)
        print('chairs', chairs)
        print('sofa', sofa)
        if is_debug:
            drawn_image = darknet.draw_boxes(detections, capture.color, class_colors)
            cv2.putText(drawn_image, 'FPS: {:.2f}'.format((1 / (time.time() - start_time))), (20, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            cv2.putText(drawn_image, 'Press \'Q\' to Exit!'.format((1 / (time.time() - start_time))), (20, 70),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            cv2.imshow('detected', drawn_image)
            if cv2.waitKey(1) in [ord('Q'), ord('q')]:
                cv2.destroyAllWindows()
                exit(0)
        count += 1
        if count >= 150:
            print(colored("Detected Failed! There is no vacant seat!"))
            return -1
        if len(persons) == 0:
            print(colored("No Person!"))
            rand_index = random.randint(1, 3)
            print(colored("Vacant seat index is {}!".format(rand_index)))
            return rand_index
        if sofa == '':
            print(colored("There is no sofa", 'bright yellow'))
            continue
        if len(chairs) == 1:
            flag, index = judge_chair_empty(persons, chairs)
            if flag and index != -1:
                print("Chair number is One!")
                print(colored("Vacant seat index is {}!".format(index)))
                return index
        if len(chairs) == 2:
            flag, index = judge_chairs_empty(persons, chairs)
            if flag and index != -1:
                print("Chair number is Two!")
                print(colored("Vacant seat index is {}!".format(index)))
                return index
        flag, index = judge_sofa_empty(persons, sofa)
        if flag and index != -1:
            print("Sofa!")
            print(colored("Vacant seat index is {}!".format(index)))
            return index


if __name__ == '__main__':
    # True代表调试模式 False代表非调试模式
    parser = argparse.ArgumentParser()
    # True代表调试模式 False代表非调试模式
    parser.add_argument('-d', '--debug', type=bool, default=True)
    args = parser.parse_args()
    is_debug = args.debug
    config_file = '/home/xm/xm_vision/darknet/cfg/yolov4.cfg'
    data_file = '/home/xm/xm_vision/darknet/cfg/coco.data'
    weights = '/home/xm/xm_vision/darknet/yolov4.weights'
    batch_size = 1
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    persons = []
    chairs = []
    sofa = ''

    camera = PyK4A()
    camera.config.color_resolution = ColorResolution.RES_1080P
    camera.start()
    capture = camera.get_capture()
    fx = width / capture.color.shape[1]
    fy = height / capture.color.shape[0]

    rospy.init_node('receptionistNode')
    service = rospy.Service('receptionist', xm_getAngle, find_vacant_seat)
    rospy.loginfo('Receptionist\'s vision start!')
    rospy.spin()
