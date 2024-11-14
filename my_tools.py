import glob
import gzip
import json
import os
import random
import re

import tqdm
import typer
from loguru import logger
from functools import cmp_to_key

app = typer.Typer()

def check(x: str):
    try:
        y = int(x)
        return True
    except ValueError:
        return False

def area(rect):
    x1, y1, x2, y2 = rect
    return (x2 - x1) * (y2 - y1)

# Custom comparison function for sorting
def custom_sort(rect1, rect2):
    area1 = area(rect1)
    area2 = area(rect2)
    
    # Check if area of rect1 is more than twice the area of rect2
    if area1 >= 2 * area2:
        return -1  # rect1 should come before rect2
    elif area2 >= 2 * area1:
        return 1  # rect2 should come before rect1
    else:
        # If areas do not satisfy the 2x condition, sort by left-to-right, top-to-bottom
        x1_1, y1_1, _, _ = rect1
        x1_2, y1_2, _, _ = rect2
        if y1_1 != y1_2:
            return y1_1 - y1_2
        else:
            return x1_1 - x1_2

def write_out(target_boxes, f_out):
    
    tmp = [(i, d) for i, d in enumerate(target_boxes)]
    random.shuffle(tmp)

    # create source from target
    target_index = [0] * len(target_boxes)
    source_boxes = []

    j = 1
    for i, _ in tmp:
        source_boxes.append(target_boxes[i].copy())
        target_index[i] = j
        j += 1

    assert(len(target_index) == len(source_boxes))

    """
    print(f"Source: {source_boxes}")
    for i in range(len(source_boxes)):
        if target_boxes[i] == source_boxes[0]:
            print(i)
            break
    print(f"Index: {target_index}")
    exit(0)
    """

    f_out.write(
        json.dumps(
            {
                "source_boxes": source_boxes,
                "source_texts": [],
                "target_boxes": target_boxes,
                "target_texts": [],
                "target_index": target_index,
                "bleu": 0.5,
            }
        )
        + "\n"
    )

@app.command()
def create_dataset_spans(
    path: str = typer.Argument(
        ...,
        help="Path to the dataset, like `./train/`",
    ),
    output_file: str = typer.Argument(
        ..., help="Path to the output file, like `./train.jsonl.gz`"
    ),
    src_shuffle_rate: float = typer.Option(
        0.5, help="The rate to shuffle input's order"
    ),
):
    random.seed(42)
    logger.info("Saving features into file {}", output_file)
    f_out = gzip.open(output_file, "wt")
    
    for label in os.listdir(path):
        
        with open(os.path.join(path, label), 'r') as f:
            data = json.load(f)

        id_to_box = {}
        target_boxes = []
        unchecked = []
        for box in data['regions']:
            name = box['shape_attributes']['name']
            if name != 'rect':
                continue
            x = box['shape_attributes']['x']
            y = box['shape_attributes']['y']
            w = box['shape_attributes']['width']
            h = box['shape_attributes']['height']
            id = box['region_attributes']['alias_id']
            if check(id) and int(id) not in id_to_box:
                id_to_box[int(id)] = [x, y, x + w, y + h]
            else:
                unchecked.append([x, y, x + w, y + h])

        # assign label for boxes were not labeled
        cnt = 1
        for x in unchecked:
            while cnt in id_to_box:
                cnt += 1
            id_to_box[cnt] = x
        id_to_box = dict(sorted(id_to_box.items()))

        for i, _ in id_to_box.items():
            target_boxes.append(id_to_box[i])

        # bbox coord compression
        max_value = 1
        for box in target_boxes:
            max_value = max(max_value, max(box))

        ratio = min(1, 1000 / max_value)
        for i in range(len(target_boxes)):
            target_boxes[i] = [min(1000, int(x * ratio)) for x in target_boxes[i]]
            assert max(target_boxes[i]) <= 1000
        
        # shuffle each page 'shuf' times to create more dataa
        shuf = 10
        for i in range(shuf):
            write_out(target_boxes, f_out)

    f_out.close()


if __name__ == "__main__":
    app()
