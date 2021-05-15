#!/usr/bin/python3
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import time
import argparse
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
        detections[i][2][1] = (height / 2 / fy) - (1 / fx) * (height / 2 - detections[i][2][1]) if detections[i][2][
                                                                                                       1] <= height / 2 \
            else (1 / fy) * (detections[i][2][1] - height / 2) + (height / 2 / fy)
        detections[i][2][2] /= fx
        detections[i][2][3] /= fy
    darknet.free_image(darknet_image)
    print(detections)
    return detections


# obj_name 即为物体的名称
# tea_pi
# cola
# spring
# juice
# soap （识别率不高 现在只能识别正面 和 斜侧面 正侧面识别困难）
def deal_with_request(obj_name):
    global is_debug
    is_found = False
    try_count = 0
    while not is_found:
        start_time = time.time()
        capture = camera.get_capture()
        detections = image_detection_original_no_image(capture.color, network, class_names,
                                                       0.4,
                                                       width / capture.color.shape[1],
                                                       height / capture.color.shape[0])
        try_count += 1
        if is_debug:
            drawn_image = darknet.draw_boxes(detections, capture.color, class_colors)
            cv2.putText(drawn_image, 'FPS: {:.2f}'.format((1 / (time.time() - start_time))), (20, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            cv2.putText(drawn_image, 'Press \'Q\' to Exit!'.format((1 / (time.time() - start_time))), (20, 70),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
        # 默认是0 如果没有识别到物体 那么返回的就是[0.0,0.0,0.0]
        coordinate = [0.0, 0.0, 0.0]
        for detection in detections:
            locals()
            x, y = int(detection[2][0]), int(detection[2][1])
            obj_coordinate = capture.transformed_depth_point_cloud[y][x]
            if is_debug:
                cv2.putText(drawn_image, '{}'.format(obj_coordinate), (x - 10, y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255, 0, 0), thickness=2)
            if detection[0] == obj_name:
                # 说明找到了
                coordinate = capture.transformed_depth_point_cloud[y][x]
                if coordinate[0] or coordinate[1] or coordinate[2]:
                    # 单位转换成 m
                    coordinate = [coordinate[0] / 1000, coordinate[1] / 1000, coordinate[2] / 1000]
                    # 说明坐标非空！ 是有效坐标！
                    is_found = True
        print(colored('{}\'s coordinate is {}'.format(obj_name, coordinate), 'bright green'))
        if try_count >= 100:
            # 说明检测了100帧也没有找到想要的物体，视为检测失败
            print(colored("Detect FAIL!!", 'red'))
            return [0, 0, 0]
        if is_debug:
            cv2.imshow("Detected Image", drawn_image)
            if cv2.waitKey(1) in [ord("q"), ord("Q")]:
                cv2.destroyAllWindows()
                exit(0)
        if is_found:
            # 这里需要改成ros通信的形式
            return coordinate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1代表调试模式 0代表非调试模式
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    is_debug = args.debug
    config_file = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/xm_vision.cfg'
    data_file = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/obj.data'
    weights = r'/home/xm/xm_vision/darknet/workspaces/XMTestWorkSpace/backup/xm_vision_last.weights'
    batch_size = 1
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    camera = PyK4A()
    camera.config.color_resolution = ColorResolution.RES_1080P
    camera.start()
    while True:
        targetCor = deal_with_request(obj_name="spring")
        print(colored('targetCor is {}'.format(targetCor), 'green'))
