import os
import csv
import cv2
import pandas as pd
import numpy as np

def image_bbox(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    cv2.imshow('thresh', thresh)
    im_floodfill = thresh.copy()

    h, w = thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0,0), (255, 255, 255))

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = thresh | im_floodfill_inv

    cnts = cv2.findContours(im_out.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    try:
        x,y,w,h = cv2.boundingRect(cnts[0])
        roi = image[y-2:y+2+h,x-2:x+w+2]

        return roi

    except:
        return image

id_to_sym_dict = {}

ocr_code_list = os.path.join(os.path.dirname(__file__), 'OcrCodeList.txt')

with open(ocr_code_list) as f:
    
    reader = csv.reader(f)

    for row in reader:

        id_hex = row[0]
        id = id_hex[2:]

        id_to_sym_dict[id] = row[2]

for id_ in id_to_sym_dict:

    ascii_id = ord(id_to_sym_dict[id_][0])

    if 97 <= ascii_id <= 122:

        if id_to_sym_dict[id_] in os.listdir(os.path.join(os.path.dirname(__file__), 'lowercase')):
            continue

        os.mkdir(os.path.join(os.path.dirname(__file__), 'lowercase', id_to_sym_dict[id_]))

    else:

        if id_to_sym_dict[id_] in os.listdir(os.path.join(os.path.dirname(__file__))):
            continue

        os.mkdir(os.path.join(os.path.dirname(__file__), id_to_sym_dict[id_]))


char_info_A_list = []

char_info_A_file = os.path.join(os.path.dirname(__file__), 'CharInfoDB-3-A_Info.csv')

with open(char_info_A_file) as f:

    reader = csv.reader(f)
    next(reader)

    for row in reader:

        char_info_A_list.append([row[1], row[2], row[3], row[4], row[5], row[6]])

char_info_B_list = []

char_info_B_file = os.path.join(os.path.dirname(__file__), 'CharInfoDB-3-B.txt')

with open(char_info_B_file) as f:

    reader = csv.reader(f)
    next(reader)

    for row in reader:

        char_info_B_list.append([row[1], row[2], row[3], row[4], row[5], row[6]])

images_A = os.path.join(os.path.dirname(__file__), 'images_A')
images_B = os.path.join(os.path.dirname(__file__), 'images_B')

images_A_files = next(os.walk(images_A))[2]
images_B_files = next(os.walk(images_B))[2]

for image_A_file in images_A_files:

    image_A_path = os.path.join(images_A, image_A_file)

    image_A = cv2.imread(image_A_path)

    sheet_name = image_A_file[:-4]

    i = 0
    for img_info in char_info_A_list:

        if img_info[1] != sheet_name:
            continue

        if img_info[0] not in id_to_sym_dict.keys():
            continue

        crop_img = image_A[(int(img_info[3]) - 40) :(int(img_info[3]) + int(img_info[4]) + 40), (int(img_info[2]) - 40) :(int(img_info[2]) + int(img_info[5]) + 40)]
        good_img = image_bbox(crop_img)
        
        ascii_id = ord(id_to_sym_dict[img_info[0]][0])

        if 97 <= ascii_id <= 122:

            cv2.imwrite(os.path.join(os.path.dirname(__file__), 'lowercase', id_to_sym_dict[img_info[0]], str(i) + '.png'), good_img)
            i += 1

        else:

            cv2.imwrite(os.path.join(os.path.dirname(__file__), id_to_sym_dict[img_info[0]], str(i) + '.png'), good_img)
            i += 1


for image_B_file in images_B_files:

    image_B_path = os.path.join(images_B, image_B_file)

    image_B = cv2.imread(image_B_path)

    sheet_name = image_B_file[:-4]

    i = 0
    for img_info in char_info_B_list:

        if img_info[1] != sheet_name:
            continue

        if img_info[0] not in id_to_sym_dict.keys():
            continue

        crop_img = image_B[(int(img_info[3]) - 10) :(int(img_info[3]) + int(img_info[4]) + 20), (int(img_info[2]) - 10) :(int(img_info[2]) + int(img_info[5]) + 20)]
        good_img = image_bbox(crop_img)

        ascii_id = ord(id_to_sym_dict[img_info[0]][0])

        if 97 <= ascii_id <= 122:

            cv2.imwrite(os.path.join(os.path.dirname(__file__), 'lowercase', id_to_sym_dict[img_info[0]], str(i) + '.png'), good_img)
            i += 1

        else:

            cv2.imwrite(os.path.join(os.path.dirname(__file__), id_to_sym_dict[img_info[0]], str(i) + '.png'), good_img)
            i += 1



        



