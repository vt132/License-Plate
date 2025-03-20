import numpy as np
import math
import cv2

def changeContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img, center_thres):
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 1.5, maxLineGap=h/3.0)
    if lines is None:
        return 1

    min_line = 100
    min_line_pos = 0
    for i in range (len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)]
            if center_thres == 1:
                if center_point[1] < 7:
                    continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    nlines = lines.size
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1
    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img, change_cons, center_thres):
    if change_cons == 1:
        angle = compute_skew(changeContrast(src_img), center_thres)
    else:
        angle = compute_skew(src_img, center_thres)
    
    rotated_img = rotate_image(src_img, angle)
    return rotated_img, angle

def transform_boxes_after_rotation(boxes, angle, image_shape):
    """
    Transform bounding boxes after image rotation
    boxes: list of [xmin, ymin, xmax, ymax, confidence, class]
    angle: rotation angle in degrees
    image_shape: (height, width) of the image
    """
    import math
    transformed_boxes = []
    
    # Calculate rotation center (image center)
    h, w = image_shape[0:2]
    cx, cy = w/2, h/2
    
    # Convert angle to radians
    angle_rad = angle * math.pi / 180
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        confidence, class_id = box[4], box[5]
        
        # Transform each corner of the bounding box
        corners = [
            [xmin, ymin], [xmax, ymin], 
            [xmax, ymax], [xmin, ymax]
        ]
        
        rotated_corners = []
        for x, y in corners:
            # Shift to origin, rotate, then shift back
            x_rot = cos_val * (x - cx) - sin_val * (y - cy) + cx
            y_rot = sin_val * (x - cx) + cos_val * (y - cy) + cy
            rotated_corners.append([x_rot, y_rot])
        
        # Get new bounding box from rotated corners
        xs = [corner[0] for corner in rotated_corners]
        ys = [corner[1] for corner in rotated_corners]
        new_xmin, new_xmax = min(xs), max(xs)
        new_ymin, new_ymax = min(ys), max(ys)
        
        transformed_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax, confidence, class_id])
    
    return transformed_boxes