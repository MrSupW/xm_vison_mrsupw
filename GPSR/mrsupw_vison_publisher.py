#!/usr/bin/python3
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import argparse
import time

import rospy
from std_msgs.msg import String, Int32, Bool, Header
from time import sleep
from geometry_msgs.msg import *
from xm_msgs.msg import *
import cv2
from pyk4a import PyK4A, ColorResolution
import sys

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


def colored(text: str, color: str = 'green') -> str:
    return "\033[" + codeCodes[color] + "m" + text + "\033[0m"


# 从所有的检测结果中筛选出来在摄像头4m范围的人
def filter_detections(detections, depth_point_cloud):
    new_detection = []
    for detection in detections:
        if detection[0].lower() == 'person' and \
                sum([(v / 1000) ** 2 for v in
                     depth_point_cloud[int(detection[2][1])][int(detection[2][0])]]) ** 0.5 <= 3.5:
            new_detection.append(detection)
    return new_detection


def which_person(filtered_detections):
    the_person = filtered_detections[0]
    for detection in filtered_detections:
        if detection[2][2] * detection[2][3] > the_person[2][2] * the_person[2][3]:
            the_person = detection
    return the_person


def are_you_sure(old_person, new_person):
    if (old_person[2][2] * old_person[2][3]) / (new_person[2][2] * new_person[2][3]) > 4:
        return False
    return True


def left_or_right(old_person):
    global width
    if old_person[2][0] < width // 2:
        return 0  # 从右边出去了
    return 1  # 从左边出去了


def people_tracking_publisher():
    global is_debug
    camera = PyK4A()
    camera.config.color_resolution = ColorResolution.RES_1080P
    camera.start()
    capture = camera.get_capture()
    fx = width / capture.color.shape[1]
    fy = height / capture.color.shape[0]
    old_person = ''
    rospy.init_node("people_tracking", anonymous=True)
    chatter_pub = rospy.Publisher("follow", xm_FollowPerson, queue_size=1)
    rate = rospy.Rate(10)
    pose_xyz = xm_FollowPerson()
    turning_right = False
    turning_left = False
    old_coordinate = [0, 0, 1000]
    while not rospy.is_shutdown():
        start_time = time.time()
        capture = camera.get_capture()
        color_image = capture.color
        detections = filter_detections(
            image_detection_original_no_image(color_image, network, class_colors, 0.7, fx, fy),
            capture.transformed_depth_point_cloud)
        coordinate = [0.0, 0.0, 0.0]
        if is_debug:
            drawn_image = color_image
        if len(detections) > 0:
            target_person = which_person(detections)
            if old_person == '':
                old_person = target_person.copy()
            print(target_person)
            if are_you_sure(old_person, target_person):
                turning_right = turning_left = False
                old_person = target_person.copy()
                coordinate = capture.transformed_depth_point_cloud[int(target_person[2][1])][int(target_person[2][0])]
                if coordinate[0] != 0.0 or coordinate[1] != 0.0 or coordinate[2] != 0.0:
                    old_coordinate = coordinate.copy()
                if is_debug:
                    cv2.rectangle(img=drawn_image, pt1=(
                        round(target_person[2][0] - target_person[2][2] / 2),
                        round(target_person[2][1] - target_person[2][3] / 2)),
                                  pt2=(round(target_person[2][0] + target_person[2][2] / 2),
                                       round(target_person[2][1] + target_person[2][3] / 2)),
                                  color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(drawn_image, 'FPS: {:.2f}'.format((1 / (time.time() - start_time))), (20, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                thickness=2)
                    cv2.putText(drawn_image, 'Press \'Q\' to Exit!'.format((1 / (time.time() - start_time))), (20, 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                                thickness=2)
                    cv2.putText(drawn_image, "{}".format(coordinate),
                                (int(target_person[2][0]), int(target_person[2][1])),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            else:
                print(colored("Person is Lost", color='purple'))
                if left_or_right(old_person):
                    turning_right = True
                    # 从右边
                    print(colored("leave from right", color='green'))
                else:
                    turning_left = True
                    # 从左边
                    print(colored("leave from left", color='blue'))
        else:
            print(colored("---------------No person-----------------", color="purple"))
            # 没有检测到人
            if old_person != '':
                # 说明上一帧还有人
                if left_or_right(old_person):
                    # 从右边
                    turning_right = True
                    print(colored("leave from right", color='green'))
                else:
                    # 从左边
                    turning_left = True
                    print(colored("leave from left", color='blue'))
        if is_debug:
            cv2.imshow("Drawn Image", drawn_image)
            if cv2.waitKey(1) in [ord('q'), ord('Q')]:
                exit(0)
        if turning_right:
            coordinate = [-12000, -12000, -12000]
        elif turning_left:
            coordinate = [-9000, -9000, -9000]
        elif len(list(coordinate)) == 0 or coordinate[0] == coordinate[1] == coordinate[2] == 0.0:
            print('coordinate = old_coordinate')
            coordinate = old_coordinate
        coordinate = [v / 1000 for v in coordinate]
        print(colored("coordinate:{}".format(coordinate), color='bright green'))
        if coordinate[0] in [-9, -12]:
            pose_xyz.position.point.x, pose_xyz.position.point.y, pose_xyz.position.point.z = coordinate
        else:
            pose_xyz.position.point.x = coordinate[2]
            pose_xyz.position.point.y = -coordinate[0]
            pose_xyz.position.point.z = -coordinate[1]
        pose_xyz.position.header.frame_id = "kinect2_rgb_link"
        pose_xyz.position.header.stamp = rospy.Time(0)  # pos_xyz.position.header.stamp = ros::Time::now();
        chatter_pub.publish(pose_xyz)
        rate.sleep()


if __name__ == '__main__':
    # True代表调试模式 False代表非调试模式
    parser = argparse.ArgumentParser()
    # True代表调试模式 False代表非调试模式
    parser.add_argument('-d', '--debug', type=bool, default=False)
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
    people_tracking_publisher()
