#!/usr/bin/env python
# -*- coding: utf-8 -*-


def check_setup(bboxes, setup):
    # print("bboxes", bboxes)
    # print("setup", setup)
    valid_setup = True
    computed_setup = {}
    valid_boxes = {}  
    for item, item_nb in setup.items():
        # print(1)
        
        # print("item", item)
        # print("item_nb", item_nb)
        if bboxes[0][item] is None:   # 这里会报key error
            valid_setup = False
        else:
            if bboxes[0][item].ndim == 2:
                box_nb = bboxes[0][item].shape[0]
                if box_nb != item_nb:
                    valid_setup = False
            else:
                box_nb = 1
            computed_setup[item] = box_nb
            valid_boxes[item] = bboxes[0][item]
        
    return valid_setup, computed_setup, valid_boxes
