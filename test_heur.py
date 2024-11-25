from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt
import json
import gzip
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from loguru import logger
import numpy as np
from typing import Dict, List
import cv2
from functools import cmp_to_key

class LayoutBlock(BaseModel):
    pred_box: list[NonNegativeFloat | NonNegativeInt | str]
    role: str
    score: float
    pred_box_pdf: list[NonNegativeFloat | NonNegativeInt | str] | None = None
    id: int

    def __repr__(self) -> str:
        return f"LayoutBlock(pred_box={self.pred_box}, role={self.role}, score={self.score}, pred_box_pdf={self.pred_box_pdf}, index={self.id})"

class Column:
    x0: int
    x1: int
    blocks: list[LayoutBlock]

    def __init__(self, x0: int, x1: int, blocks: list[LayoutBlock]):
        self.x0 = x0
        self.x1 = x1
        self.blocks = blocks

def sort_blocks(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    """Sort blocks into columns based on the horizontal overlap.

    This function aims to manage the complexity in switching between single column
    and multi-column layout. It tries to sort the blocks as accurate as possible,
    but it is not perfect.
    """
    if len(blocks) == 0:
        return []
    min_x0 = min([block.pred_box[0] for block in blocks])
    max_x1 = max([block.pred_box[2] for block in blocks])
    page_width = max_x1 - min_x0

    columns: list[Column] = []
    sorted_blocks = []
    is_single_column = True

    blocks.sort(key=lambda x: x.pred_box[1])
    for block in blocks:
        overlapping_columns = []
        for column in columns:
            # Check if the block is in the column based on the horizontal overlap
            if block.pred_box[0] >= column.x1 or block.pred_box[2] <= column.x0:
                continue
            column.blocks.append(block)
            column.x0 = min(column.x0, block.pred_box[0])
            column.x1 = max(column.x1, block.pred_box[2])
            overlapping_columns.append(column)

        if overlapping_columns:
            if len(overlapping_columns) > 1:
                # block than span multiple columns means the layout switch from multi- to single-column
                columns.sort(
                    key=lambda x: x.x0
                )  # Sort list of columns left to right

                # flatten all blocks in columns, finally add the spanned block
                for column in columns:
                    if block in column.blocks:
                        sorted_blocks.extend(column.blocks[:-1])
                    else:
                        sorted_blocks.extend(column.blocks)

                sorted_blocks.append(block)
                columns = []  # reset buffer
                is_single_column = True
        else:
            if (
                is_single_column
                and block.pred_box[2] - block.pred_box[0] > 0.5 * page_width
            ):
                # block wider than half the page is considered as a single column
                sorted_blocks.append(block)
            else:
                # Start new column
                columns.append(
                    Column(
                        x0=block.pred_box[0], x1=block.pred_box[2], blocks=[block]
                    )
                )
                if len(columns) > 1:
                    # now it is multi-column
                    is_single_column = False
    
    for column in columns:
        column.blocks.sort(key=lambda x: x.pred_box[1])
    columns.sort(key=lambda x: x.x0)
    sorted_blocks.extend([block for column in columns for block in column.blocks])

    return sorted_blocks

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

def heur_flax_predict(source_boxes):
    boxes = [{'location': [[box[0], box[1]], [box[2], box[3]]], 'id': i} for i, box in enumerate(source_boxes)]
    # print(boxes)
    boxes = sort_textlines(boxes)
    res = []
    for box in boxes:
        res.append(box['id'])
    return res

if __name__ == '__main__':
    input_file = "./cinbank/test.jsonl.gz" 

    datasets = []
    with gzip.open(input_file, "rt") as f:
        for line in tqdm(f):
            datasets.append(json.loads(line))

    logger.info(len(datasets))

    total_out_idx = 0
    total = 0
    
    for i in tqdm(range(0, len(datasets))): 
        data = datasets[i]
        """
        boxes = []
        for ind in range(len(data["source_boxes"])):
            boxes.append(LayoutBlock(pred_box=data["source_boxes"][ind], role="a", score=1.0, pred_box_pdf=data["source_boxes"][ind], id=ind))
        sorted_block = sort_blocks(boxes)
        pred_index = [x.id for x in sorted_block]
        """

        pred_index = heur_flax_predict(data["source_boxes"])
        chen_cherry = SmoothingFunction()
        total += 1
        total_out_idx += sentence_bleu(
                [data["target_index"]],
                [i + 1 for i in pred_index],
                smoothing_function=chen_cherry.method2,
            )

    print("out_idx: ", round(100 * total_out_idx / total, 1))