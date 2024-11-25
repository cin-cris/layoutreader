import cv2
import pymupdf
import requests
import json
from typing import List, Dict
from functools import cmp_to_key
import numpy as np
from loguru import logger

def model_predict(boxes, maxcor):
    # send to server to predict orders
    r = requests.post(
        "http://localhost:8000/predict",
        json={"boxes": boxes, "width": maxcor, "height": maxcor},
    )

    orders = r.json()["orders"]
    print(orders)
    # reorder boxes
    boxes = [boxes[i] for i in orders]
    return boxes, orders

def is_horizontal_overlap(
    loc1: List[tuple],
    loc2: List[tuple],
    thres: float = 0.3,
    vertical_offset: float = 0.0,
) -> bool:
    """Description: Verifying horizontal alignment between 2 locations
    Args:
        loc1 (List[tuple]): input location 1
        loc2 (List[tuple]): input location 2
        thres (float): tolerant threshold for decision
        vertical_offset (float): additional offset of second location in this alignment
            (calculate relatively with location height)

    Returns:
        Return True if horizontal overlapping, otherwise return False
    """
    # get first location attributes
    x1_1, y1_1, w1, h1 = cv2.boundingRect(np.array(loc1))
    # x1_2 = x1_1 + w1  # TODO: 'x1_2' is assigned to but never used
    y1_2 = y1_1 + h1

    # get second location attributes
    x2_1, y2_1, w2, h2 = cv2.boundingRect(np.array(loc2))
    # x2_2 = x2_1 + w2  # TODO: 'x2_2' is assigned to but never used
    y2_2 = y2_1 + h2

    # add offset to second location attributes
    y2_1 = y2_1 + vertical_offset * h2
    y2_2 = y2_2 + vertical_offset * h2

    # verify by horizontal intersection
    h_intersect = max(0, min(y1_2, y2_2) - max(y2_1, y1_1))
    isHorizontal = h_intersect > min(h1, h2) * thres

    return isHorizontal

def compare_location(textline1: Dict, textline2: Dict) -> int:
    """Ordering 2 textlines by top-bottom, left-right
    Args:
        textline1 (Dict): input textline 1 with location information
        textline2 (Dict): input textline 2 with location information
    Returns:
        The difference between the left coordinate of textline1 and textline2 if they overlap horizontally,
             otherwise the difference between the top coordinate of textline1 and textline2.
    Examples:
        >>> textline1 = {'location': [10, 20, 100, 50]}
        >>> textline2 = {'location': [20, 40, 80, 30]}
        >>> compare_location(textline1, textline2)
        -10
    """
    # get textline location attributes
    #logger.info(np.array(textline1.get("location")).astype(int))
    x1, y1, w1, h1 = cv2.boundingRect(np.array(textline1.get("location")).astype(int))
    x2, y2, w2, h2 = cv2.boundingRect(np.array(textline2.get("location")).astype(int))

    # compare textline location
    if is_horizontal_overlap(
        textline1.get("location"), textline2.get("location"), thres=0.5
    ):
        return x1 - x2

    else:
        return y1 - y2

def sort_textlines(layout_output: List[Dict]) -> List[Dict]:
    """Sorts a list of textlines by their location.

    Args:
        layout_output (list[dict]): A list of textlines follow std lib-layout format

    Returns:
        A sorted list of textlines, ordered from top to bottom and left to right.

    """
    return sorted(layout_output, key=cmp_to_key(compare_location))

def heur_predict(boxes):
    boxes = [{'location': [[box[0], box[1]], [box[2], box[3]]]} for box in boxes]
    # print(boxes)
    boxes = sort_textlines(boxes)
    res = []
    for box in boxes:
        res.append(box['location'][0] + box['location'][1])
    return res, []

# Please `python main.py` first

file_name = "POL1013BA_SBD_33_100"
page_img_file = f"./aur-684_multimodal_document_parsing_v2/input/{file_name}.png"
label = f"./aur-684_multimodal_document_parsing_v2/label/{file_name}.json"

with open(label, 'r') as f:
    data = json.load(f)

boxes = []
maxcor = 0

for box in data['regions']:
    name = box['shape_attributes']['name']
    if name != 'rect':
        continue
    x = box['shape_attributes']['x']
    y = box['shape_attributes']['y']
    w = box['shape_attributes']['width']
    h = box['shape_attributes']['height']
    boxes.append([x, y, x + w, y + h])
    maxcor = max(maxcor, max(x + w, y + h))

boxes, orders = model_predict(boxes, maxcor)
#boxes, orders = heur_predict(boxes)

# draw boxes
img = cv2.imread(page_img_file)
for idx, box in enumerate(boxes):
    x0, y0, x1, y1 = box
    x0 = round(x0)
    y0 = round(y0)
    x1 = round(x1)
    y1 = round(y1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
    cv2.putText(
        img,
        str(idx),
        (x1, y1),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 0, 0),
        1,
    )
logger.info(f"SUCCESSFUL to file {file_name}.png")
cv2.imwrite(f"./{file_name}.png", img)
