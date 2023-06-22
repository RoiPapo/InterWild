

import json
import glob
import numpy as np
import cv2
import os
import tqdm



def draw_bounding_box_yolo(image, bbox_yolo):
    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Extract class and bounding box coordinates
    class_id = int(bbox_yolo[0])
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_yolo[1:]

    # Convert YOLO bounding box to absolute coordinates
    bbox_x = int(bbox_x * image_width)
    bbox_y = int(bbox_y * image_height)
    bbox_w = int(bbox_w * image_width)
    bbox_h = int(bbox_h * image_height)

    # Calculate bounding box coordinates
    bbox_xmin = bbox_x - bbox_w // 2
    bbox_ymin = bbox_y - bbox_h // 2
    bbox_xmax = bbox_xmin + bbox_w
    bbox_ymax = bbox_ymin + bbox_h

    # Draw bounding box on the image
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    cv2.rectangle(image, (bbox_xmin, bbox_ymin), (bbox_xmax, bbox_ymax), color, thickness)

    # Add class label to the bounding box
    label = f"Class: {class_id}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(image, (bbox_xmin, bbox_ymin - label_size[1] - 10),
                  (bbox_xmin + label_size[0], bbox_ymin - 10), color, cv2.FILLED)
    cv2.putText(image, label, (bbox_xmin, bbox_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return image




def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]


def parse_whole_json_file(path):
    id_filename_dict={}
    with open(path) as f:
        data = json.load(f)
    locs = data['images']
    annotations= data['annotations']
    for rowA,rowB  in  zip(locs,annotations):
        if rowA["id"]==rowB["id"]:
            id_filename_dict[rowA["file_name"]] = rowB["bbox"]
        else:
            print("Error",rowA["id"]," != ", rowB["id"] )
    return id_filename_dict

            

def create_ds(images_source,box_dict,export_folder):
    file_count = sum(len(files) for _, _, files in os.walk(images_source))  # Get the number of files
    pbar = tqdm.tqdm(total=file_count)
    for root, directories, files in os.walk(images_source):
        for file in files:
            file_path = os.path.join(root, file)
            image = cv2.imread(file_path)
            height, width = image.shape[:2]
            local_path= file_path.split('/train/')[-1]
            bbox= [0]+coco_to_yolo(*box_dict[local_path],width,height)
            image_name= local_path.replace('/', '_')
            with open(export_folder+'/labels/'+image_name[:-4]+".txt", 'w') as file:
                file.write("%s " % bbox)
            cv2.imwrite(export_folder+'/images/'+image_name, image)
            pbar.update(1)
            # img_box = draw_bounding_box_yolo(image, bbox)
            # cv2.imwrite(export_folder+'/images/box_'+image_name, img_box)
            #hi
            



image = cv2.imread("/data/home/roipapo/InterWild/data/InterHand26M/images/train/Capture18/0389_dh_nontouchROM/cam400002/image54151.jpg")

with open('/data/home/roipapo/HandsDetection/images/interhand26M/train/labels/Capture18_0389_dh_nontouchROM_cam400002_image54151.txt', "r") as file:
    bbox = json.load(file)

x= draw_bounding_box_yolo(image,bbox)
cv2.imwrite("stuff.png", x)
exit()




images_source ="/data/home/roipapo/InterWild/data/InterHand26M/images/train"
whole_file_path="/data/home/roipapo/InterWild/data/InterHand2.6M_train_data.json"
box_dict = parse_whole_json_file(whole_file_path)
export_folder ="/data/home/roipapo/HandsDetection/images/interhand26M/train"
create_ds(images_source,box_dict,export_folder)



