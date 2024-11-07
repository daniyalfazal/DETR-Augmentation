import os  
import random  
import json  
import numpy as np  
from PIL import Image  
import albumentations as A  
from pycocotools.coco import COCO 

# Paths to the 1% dataset and COCO files (modify these paths based on where you have the dataset stored)
one_percent_images_directory = 'C:/coco_subset_20_classes/train2017'  
one_percent_annotations_path = 'C:/coco_subset_20_classes/annotations/instances_train2017.json'  
augmented_images_directory = 'C:/coco_subset_20_classes/augmentedtrain2017'  
augmented_annotations_path = 'C:/coco_subset_20_classes/annotations/instances_augmentedtrain2017.json'

# Create output directory if it doesn't exist
os.makedirs(augmented_images_directory, exist_ok=True) 

# Load the annotations for the 1% dataset that we are using
with open(one_percent_annotations_path, 'r') as f: 
    one_pct_data = json.load(f)

# Performing the Augmentation techniques on the dataset being used. (Effect of each technique is further explained in detail in the paper)
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2), 
    A.Rotate(limit=30),
    A.GaussNoise(var_limit=(10.0, 50.0)), 
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),  
])

# Saving a copy of the annotations file to be able to append augmented data to it
aug_annotations = one_pct_data.copy() 

# Augmenting each image 
for img_info in one_pct_data['images']: 
    img_id = img_info['id']  
    img_name = img_info['file_name'] 
    img_path = os.path.join(one_percent_images_directory, img_name)

    # Check to see if the image actually exists
    if not os.path.exists(img_path):  
        print(f"Image file {img_path} not found. Skipping this image.")  
        continue

    # Opening each image using PIL
    with Image.open(img_path) as img:
        img_np = np.array(img)

        
        for i in range(5):  
            # Applying each augmentation defined above
            augmented = augmenter(image=img_np) 
            aug_img_np = augmented['image']

            
            aug_img_pil = Image.fromarray(aug_img_np)

           
            new_img_name = f"{img_id:012d}_aug_{i}.jpg" 
            new_img_path = os.path.join(augmented_images_directory, new_img_name) 
            aug_img_pil.save(new_img_path) 

            new_img_id = int(f"{img_id}{i}")  
            new_img_info = {
                "id": new_img_id,  
                "file_name": new_img_name, 
                "width": img_info['width'], 
                "height": img_info['height'] 
            }
            aug_annotations['images'].append(new_img_info)  

            # Looking for the annotation associated with this image
            related_annotations = [
                ann for ann in one_pct_data['annotations'] if ann['image_id'] == img_id 
            ]

            
            for ann in related_annotations:  
                new_ann = ann.copy()
                new_ann['id'] = int(f"{ann['id']}{i}")
                new_ann['image_id'] = new_img_id 
                aug_annotations['annotations'].append(new_ann) 


with open(augmented_annotations_path, 'w') as f: 
    json.dump(aug_annotations, f)  

# Output a message that indicates that the task has been completed. 
print(f"Augmented images and annotations saved to {augmented_images_directory} and {augmented_annotations_path}.")
