import cv2
import pymupdf
import requests
import json

# Please `python main.py` first

file_name = "Presentation_2024_4QJ_30_55"
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

# send to server to predict orders
r = requests.post(
    "http://localhost:8000/predict",
    json={"boxes": boxes, "width": maxcor, "height": maxcor},
)

orders = r.json()["orders"]
print(orders)
# reorder boxes
boxes = [boxes[i] for i in orders]
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
cv2.imwrite("./predicted.png", img)
