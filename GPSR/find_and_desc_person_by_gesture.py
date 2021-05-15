#!/usr/bin/python3
# -*- coding: utf-8 -*-
# [centerX,centerY,width,height]
import argparse
import threading

import cv2
from aip import AipBodyAnalysis
from pyk4a import PyK4A, ColorResolution

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
analysis_result = []
attr_result = []
gestures = ['waving', 'left_point', 'right_point', 'left_raise', 'right_raise']
POINT_X_THRESHOLD = 100
POINT_Y_THRESHOLD = 50
HANDS_UP_THRESHOLD = 50


def colored(text, color='green'):
    return "\033[" + codeCodes[color] + "m" + str(text) + "\033[0m"


def drawDebugImage(image, final_results):
    for final_result in final_results:
        location = final_result['location']
        cv2.rectangle(image, (round(location['left']), round(location['top'])), (round(location['left'] + location['width']), round(location['top'] + location['height'])), (0, 255, 0), thickness=2)
        cv2.circle(image, final_result['right_wrist'], 2, (255, 255, 0), thickness=2)
        cv2.circle(image, final_result['right_elbow'], 2, (255, 255, 0), thickness=2)
        cv2.circle(image, final_result['right_shoulder'], 2, (255, 255, 0), thickness=2)
        cv2.circle(image, final_result['left_wrist'], 2, (0, 255, 255), thickness=2)
        cv2.circle(image, final_result['left_elbow'], 2, (0, 255, 255), thickness=2)
        cv2.circle(image, final_result['left_shoulder'], 2, (0, 255, 255), thickness=2)
    return image


def deal_with_bodyAttr_result(img_bytes):
    global attr_result
    attr_result = client.bodyAttr(img_bytes)
    if attr_result["person_num"] > 0:
        result_data = []
        for person_info in attr_result['person_info']:
            person_data = {}
            attributes = person_info["attributes"]
            person_data["gender"] = attributes["gender"]["name"]
            person_data["age"] = attributes["age"]['name']
            person_data["upper_color"] = attributes["upper_color"]['name']
            person_data["glasses"] = attributes["glasses"]['name']
            person_data["face_mask"] = attributes["face_mask"]['name']
            person_data["headwear"] = attributes["face_mask"]['name']
            person_data["headwear"] = attributes["headwear"]['name']
            person_data["location"] = person_info["location"]
            person_data["center"] = (round(person_data["location"]["left"] + person_data["location"]["width"] / 2),
                                     round(person_data["location"]["top"] + person_data["location"]["height"] / 2))
            result_data.append(person_data)
        attr_result = result_data
    else:
        attr_result = []


def deal_with_bodyAnalysis_result(img_bytes):
    global analysis_result
    analysis_result = client.bodyAnalysis(img_bytes)
    if analysis_result["person_num"] > 0:
        result_data = []
        for person_info in analysis_result['person_info']:
            person_data = {}
            body_parts = person_info["body_parts"]
            person_data["right_wrist"] = (round(body_parts["right_wrist"]["x"]), round(body_parts["right_wrist"]["y"]))
            person_data["right_elbow"] = (round(body_parts["right_elbow"]["x"]), round(body_parts["right_elbow"]["y"]))
            person_data["right_shoulder"] = (round(body_parts["right_shoulder"]["x"]), round(body_parts["right_shoulder"]["y"]))
            person_data["left_wrist"] = (round(body_parts["left_wrist"]["x"]), round(body_parts["left_wrist"]["y"]))
            person_data["left_elbow"] = (round(body_parts["left_elbow"]["x"]), round(body_parts["left_elbow"]["y"]))
            person_data["left_shoulder"] = (round(body_parts["left_shoulder"]["x"]), round(body_parts["left_shoulder"]["y"]))
            person_data["location"] = person_info["location"]
            person_data["center"] = (round(person_data["location"]["left"] + person_data["location"]["width"] / 2),
                                     round(person_data["location"]["top"] + person_data["location"]["height"] / 2))
            gesture_index = judge_gesture(person_data)
            if gesture_index != -1:
                person_data['gesture'] = gestures[gesture_index]
                person_data['gesture_index'] = gesture_index
            else:
                person_data['gesture'] = 'idle'
                person_data['gesture_index'] = -1
            result_data.append(person_data)
        analysis_result = result_data
    else:
        analysis_result = []


def integrate_results(attr_results, analysis_results):
    final_result = []
    for a_result in attr_results:
        for as_result in analysis_results:
            if abs(a_result["center"][0] - as_result["center"][0]) < 50:
                t_result = as_result.copy()
                t_result["gender"] = a_result["gender"]
                t_result["age"] = a_result["age"]
                t_result["upper_color"] = a_result["upper_color"]
                t_result["glasses"] = a_result["glasses"]
                t_result["face_mask"] = a_result["face_mask"]
                t_result["headwear"] = a_result["headwear"]
                final_result.append(t_result)
                break
    return final_result


def judge_gesture(person_data):
    # 判断是否是向左指
    if abs(person_data['left_wrist'][1] - person_data['left_elbow'][1]) < POINT_Y_THRESHOLD and \
            abs(person_data['left_wrist'][1] - person_data['left_shoulder'][1]) < POINT_Y_THRESHOLD and \
            abs(person_data['left_wrist'][0] - person_data['left_shoulder'][0]) > POINT_X_THRESHOLD:
        return 1
    # 判断是否是向右指
    if abs(person_data['right_wrist'][1] - person_data['right_elbow'][1]) < POINT_Y_THRESHOLD and \
            abs(person_data['right_wrist'][1] - person_data['right_shoulder'][1]) < POINT_Y_THRESHOLD and \
            abs(person_data['right_wrist'][0] - person_data['right_shoulder'][0]) > POINT_X_THRESHOLD:
        return 2
    # 判断是否举左手
    if person_data['left_elbow'][1] - person_data['left_wrist'][1] > HANDS_UP_THRESHOLD:
        return 3
    # 判断是否举右手
    if person_data['right_elbow'][1] - person_data['right_wrist'][1] > HANDS_UP_THRESHOLD:
        return 4
    return -1


def find_person_by_gesture(results, ges_index):
    if ges_index == -1:
        return results[0]
    if ges_index == 0:
        for result in results:
            if result['gesture_index'] in [3, 4]:
                return result
    for result in results:
        if result['gesture_index'] == ges_index:
            return result
    return None


def generate_description(result):
    if result['gender'] == '男性':
        desc = "He is {glasses}wearing glasses. He is {mask}wearing a face mask. He is {headwear}wearing a hat."
    else:
        desc = "She is {glasses}wearing glasses. She is {mask}wearing a face mask. She is {headwear}wearing a hat."
    desc = desc.format(glasses='' if result['glasses'] == '戴眼镜' else 'not ',
                       mask='' if result['face_mask'] == '戴口罩' else 'not ',
                       headwear='' if result['headwear'] in ['普通帽', '安全帽'] else 'not ')
    return desc


if __name__ == "__main__":
    APP_ID = '23757775'
    API_KEY = 'K0uRowPaoSfSs0ABvUxurnHG'
    SECRET_KEY = 'Q6WtWCSFqqMAbKffCrqHQ6DDWRNc0oD7'
    client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=bool, default=False)
    args = parser.parse_args()
    is_debug = args.debug
    camera = PyK4A()
    camera.config.color_resolution = ColorResolution.RES_1080P
    camera.start()
    c_type = 1
    gesture = 0
    print('starting...')
    target_gesture = 3
    while True:
        capture = camera.get_capture()
        img_bytes = cv2.imencode('.jpg', capture.color)[1].tobytes()
        attr_thread = threading.Thread(target=deal_with_bodyAttr_result, args=(img_bytes,))
        analysis_thread = threading.Thread(target=deal_with_bodyAnalysis_result, args=(img_bytes,))
        attr_thread.start()
        analysis_thread.start()
        attr_thread.join()
        analysis_thread.join()
        final_results = integrate_results(attr_result, analysis_result)
        if len(final_results) == 0:
            continue
        print(colored(final_results))
        drawn_image = drawDebugImage(capture.color, final_results)
        if is_debug:
            cv2.imshow("Detected Image", drawn_image)
            if cv2.waitKey(1) in [ord("q"), ord("Q")]:
                cv2.destroyAllWindows()
                exit(0)
        result = find_person_by_gesture(final_results, target_gesture)
        if result:
            description = generate_description(result)
            coordinate = capture.transformed_depth_point_cloud[result['center'][1], result['center'][0]]
            coordinate = [v / 1000 for v in coordinate]
            print(colored(description, 'bright purple'))
            print(colored("coordinate:{}".format(coordinate)))
        else:
            print(colored("No qualified person!", 'bright red'))
