#!/usr/bin/python3
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import argparse
from aip import AipImageClassify
import rospy
from pyk4a import PyK4A
from std_msgs.msg import String, Int32, Bool, Header
from geometry_msgs.msg import *
from xm_msgs.msg import *
from xm_msgs.srv import *
import cv2

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

names = {"家居家纺": 'Textiles', '果蔬生鲜': 'Fruits',
         "食品饮料": 'Food', "家居家电": "Appliances",
         "文化娱乐": 'Entertainment', '鞋品箱包': "Shoes",
         "电子数码": "Electronic", '服装饰品': "Cloth",
         "人": "Person", "运动户外": "Sports", }


def colored(text, color='green'):
    return "\033[" + codeCodes[color] + "m" + text + "\033[0m"


def drawDebugImage(image, result):
    for res in result:
        loc = res['location']
        cv2.putText(image, names.get(res['name'], 'unknown'), org=(loc['left'] + loc['width'] // 2 - 20, loc['top'] + loc['height'] // 2 - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                    color=(255, 255, 0), thickness=1,
                    lineType=cv2.LINE_AA)
        cv2.rectangle(image, (loc['left'], loc['top']), (loc['left'] + loc['width'], loc['top'] + loc['height']), (0, 255, 255), 2, lineType=cv2.LINE_AA)
    return image


def deal_with_request(classification, type):
    global is_debug
    count = 0
    is_found = False
    coordinate = [0, 0, 0]
    while True:
        count += 1
        capture = camera.get_capture()
        res = client.multiObjectDetect(cv2.imencode('.jpg', capture)[1].tobytes())
        if type:
            # Pick Up
            for r in res['result']:
                if names.get(r['name']) == classification:
                    # 获取位置
                    center = (r['location']['left'] + r['location']['width'] // 2, r['location']['top'] + r['location']['height'] // 2)
                    coordinate = capture.transformed_depth_point_cloud[center[1], center[0]]
                    coordinate[0] /= 1000
                    coordinate[1] /= 1000
                    coordinate[2] /= 1000
                    if coordinate[0] or coordinate[1] or coordinate[2]:
                        is_found = True
                        print(colored("{}'s coordinate is {}".format(classification, coordinate)))
                        break
        else:
            # Put Down

            pass
        if is_found:
            return coordinate
        if count >= 5:
            print(colored("NOT FOUND!", 'bright red'))
            return coordinate


def call_back(obj):  # req的数据类型是xm_ObjectDetect()
    res = xm_ObjectDetectResponse()  # 数据类型相当于xm_ObjectDetect
    classification = obj.classification
    type = obj.type
    targetCor = deal_with_request(classification, type)
    res.object.pos.point.x = targetCor[2]
    res.object.pos.point.y = -targetCor[0]
    res.object.pos.point.z = -targetCor[1]
    res.object.pos.header.frame_id = "kinect2_rgb_link"
    # res.pos.header.frame_id = "camera_body"
    # res.pos.header.frame_id = "camera_visor"
    res.object.pos.header.stamp = rospy.Time(0)
    res.object.state = 1
    return res


if __name__ == "__main__":
    APP_ID = '24052863'
    API_KEY = 'fKa0YNxEzx4AzeN26K0HqfGZ'
    SECRET_KEY = '1LZ37nxW8WZGq07AKGZBZaZ5SmIhBuNY'
    client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)
    # True代表调试模式 False代表非调试模式
    parser = argparse.ArgumentParser()
    # True代表调试模式 False代表非调试模式
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    is_debug = args.debug
    camera = PyK4A()
    camera.start()
    first_capture = camera.get_capture()
    rospy.init_node('object_detect')
    service = rospy.Service('get_position', xm_ObjectDetect, call_back)
    rospy.loginfo('object_detect')
    print("time out")
    rospy.spin()
