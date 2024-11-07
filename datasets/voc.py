import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset

class PascalVOCDataset(Dataset):
    def __init__(self, root, image_set='train', transforms=None):
        self.root = os.path.join(root, "VOCdevkit", "VOC2012")
        self.transforms = transforms
        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.annotations_dir = os.path.join(self.root, "Annotations")
        
        # Load the file names of images for the specified set (train or val)
        with open(os.path.join(self.root, "ImageSets", "Main", f"{image_set}.txt")) as f:
            self.file_names = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        image_id = self.file_names[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        annotation_path = os.path.join(self.annotations_dir, f"{image_id}.xml")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations (Pascal VOC uses XML format)
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        area = []
        iscrowd = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            labels.append(PascalVOCDataset.class_dict[label])
            
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            
            # VOC dataset doesn't have 'iscrowd', set all to 0
            iscrowd.append(0)
            
            # Calculate area of the bounding box
            area.append((xmax - xmin) * (ymax - ymin))
        
        # Convert everything to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([int(image_id)])
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # Apply transformations (only to image)
        if self.transforms is not None:
            img = self.transforms(img)  # Apply transforms to the image only
        
        return img, target


# Add a dictionary for Pascal VOC class labels to integers
PascalVOCDataset.class_dict = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
    'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}