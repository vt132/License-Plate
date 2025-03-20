import cv2
import torch
import numpy as np
import math
from .function import helper
from .function import utils_rotate

from torch import functional as F
from spandrel import ModelLoader



# Load model
model_path = '8x_NMKD-Typescale_175k.pth'
model = ModelLoader().load_from_file(model_path)
denoise_strength  = 1
upsampler = helper.ESRGANer(
    scale=8,
    model=model,
    tile=0,
    pre_pad=0,
    tile_pad=10,
    dni_weight = [denoise_strength, 1 - denoise_strength],
    device='cuda',
)
denoise_strength = 1

# Initialize the Real-ESRGAN enhancer
# Load the YOLOv5 models for license plate detection and text recognition
yolo_LP_detect = torch.hub.load(
    'yolov5',
    'custom',
    path='license_plate.pt',
    force_reload=True,
    source='local',
)
yolo_license_plate = torch.hub.load(
    'yolov5',
    'custom',
    path='letter_detection.pt',
    force_reload=True,
    source='local',
)
def detect_license_plate(img, upscale=True):
    plates = yolo_LP_detect(img, size=640)
    print(plates)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            x = int(plate[0])  # xmin
            y = int(plate[1])  # ymin
            w = int(plate[2] - plate[0])  # xmax - xmin
            h = int(plate[3] - plate[1])  # ymax - ymin
            crop_img = img[y:y+h, x:x+w]
            
            if upscale:
                enhanced_img, _ = upsampler.enhance(crop_img)
                plate_img = enhanced_img
            else:
                plate_img = crop_img
            
            # Step 1: Detect characters in original image first
            char_results = yolo_license_plate(plate_img, size=640)
            char_boxes = char_results.pandas().xyxy[0].values.tolist()
            print(char_results)
            print(char_boxes)
            # Step 2: If enough characters detected, try deskewing and transform coordinates
            if len(char_boxes) >= 7 and len(char_boxes) <= 10:
                # Try deskewing with coordinate transformation
                for cc in range(0, 2):
                    for ct in range(0, 2):
                        deskewed_img, angle = utils_rotate.deskew(plate_img, cc, ct)
                        transformed_boxes = utils_rotate.transform_boxes_after_rotation(
                            char_boxes, angle, plate_img.shape)
                        
                        lp = helper.read_plate(yolo_license_plate, deskewed_img, 
                                              pre_detected_boxes=transformed_boxes)
                        
                        if lp != "unknown":
                            list_read_plates.add(lp)
                            cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            break
                    if lp != "unknown":
                        break
            
            # Step 3: If not enough characters, try deskewing and detecting again
            else:
                for cc in range(0, 2):
                    for ct in range(0, 2):
                        deskewed_img, _ = utils_rotate.deskew(plate_img, cc, ct)
                        
                        # Normal detection on deskewed image (no coordinate transform)
                        lp = helper.read_plate(yolo_license_plate, deskewed_img)
                        
                        if lp != "unknown":
                            list_read_plates.add(lp)
                            cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            break
                    if lp != "unknown":
                        break
    
    print(list_read_plates)
    return list_read_plates


def detect_license_plate_enhanced(img, upscale=True):
    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            x = int(plate[0])  # xmin
            y = int(plate[1])  # ymin
            w = int(plate[2] - plate[0])  # xmax - xmin
            h = int(plate[3] - plate[1])  # ymax - ymin
            crop_img = img[y:y+h, x:x+w]
            if upscale:
                enhanced_img, _ = upsampler.enhance(crop_img)
                return enhanced_img
            return crop_img
    return None


