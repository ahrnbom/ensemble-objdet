""" 
Ensembling methods for object detection.
"""

""" 
Basic Ensemble - find overlapping boxes of the same class and average their positions
while adding their confidences

Input: 
 - dets : List of detections. Each detection is the output from one detector, and
          should be a list of boxes, where each box should be on the format 
          [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y 
          are the center coordinates, box_w and box_h are width and height resp.
          The values should be floats, except the class which should be an integer.

 - iou_thresh: Threshold in terms of IOU where two boxes are considered the same, 
               if they also belong to the same class.

Output:
    A list of boxes, on the same format as the input. Confidences are in range 0-1.
"""
def BasicEnsemble(dets, iou_thresh = 0.5):
    ndets = len(dets)
    out = list()

    used = dict()
    
    for det in dets:
        other_dets = [d for d in dets if d != det] 
        for box in det:
            if box in used:
                continue
                
            used[box] = 1
            # Search the other detectors for overlapping box of same class
            found = []
            for odet in other_dets:
                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox
                
                if not bestbox is None:
                    found.append(bestbox)
                    used[bestbox] = 1
                            
            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [box]
                allboxes.extend(found)
                
                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0
                
                for b in allboxes:
                    xc += b[0]
                    yc += b[1]
                    bw += b[2]
                    bh += b[3]
                    conf += b[5]
                    
                nall = len(allboxes)
                xc /= nall
                yc /= nall
                bw /= nall
                bh /= nall
                conf /= ndets
                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out
    
def getCoords(box):
    x1 = box[0] - box[2]/2
    x2 = box[0] + box[2]/2
    y1 = box[1] - box[3]/2
    y2 = box[1] + box[3]/2
    return x1, x2, y1, y2
    
def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)
    
    x_left   = max(x11, x21)
    y_top    = max(y11, y21)
    x_right  = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0    
        
    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)        
    
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou
