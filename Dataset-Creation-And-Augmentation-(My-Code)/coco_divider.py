import json
import os
import shutil
import random

#Please note that this script can only do one folder at a time, so you would have to run it twice. Once for the
#train folder and once for the Val folder.

# Path to the original COCO annotations file (Modify according to your actual path where the data is stored)
coco_annotation_file = 'C:/coco/annotations_trainval2017/instances_train2017.json'

# Path to save the new subset annotations file (Modify according to your actual path where the data is stored)
subset_annotation_file = 'C:/coco/instances_train2017_subset.json'

# Path to the original COCO images folder (Modify according to your actual path where the data is stored)
coco_images_directory = 'C:/coco/train2017/'

# Path to save the subset of images (Modify according to your actual path where the data is stored)
subset_images_directory = 'C:/coco/subset/train2017/'
os.makedirs(subset_images_directory, exist_ok=True)

# Define the categories you want to keep (adjust this as needed)
chosen_categories = [77, 76, 66, 64, 42, 40, 43, 44, 45, 46, 1, 2, 3, 5, 6, 7, 9, 58, 60, 61]


with open(coco_annotation_file, 'r') as f:
    coco_data = json.load(f)


new_categories = [cat for cat in coco_data['categories'] if cat['id'] in chosen_categories]


total_images = len(coco_data['images'])
one_percent_images = int(0.01 * total_images)


category_to_images = {cat['id']: [] for cat in new_categories}
for ann in coco_data['annotations']:
    if ann['category_id'] in chosen_categories:
        category_to_images[ann['category_id']].append(ann['image_id'])

# Randomly selecting images from each chosen category, distributing 1% across the 20 categories
selected_image_ids = set()
images_per_category = one_percent_images // len(chosen_categories)

for category, image_ids in category_to_images.items():
    if len(image_ids) > images_per_category:
        selected_image_ids.update(random.sample(image_ids, images_per_category))
    else:
        selected_image_ids.update(image_ids)  # In case there are fewer images than needed


new_images = [img for img in coco_data['images'] if img['id'] in selected_image_ids]
new_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in selected_image_ids]

# Copy the selected images to the new folder for the subset
for img in new_images:
    image_filename = img['file_name']
    src_image_path = os.path.join(coco_images_directory, image_filename)
    dest_image_path = os.path.join(subset_images_directory, image_filename)
    
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dest_image_path)

# Create the new COCO annotation format
new_coco_data = {
    'images': new_images,
    'annotations': new_annotations,
    'categories': new_categories
}


with open(subset_annotation_file, 'w') as f:
    json.dump(new_coco_data, f)

print(f"Subset annotations and images saved successfully.")