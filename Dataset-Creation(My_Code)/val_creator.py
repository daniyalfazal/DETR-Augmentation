import os
import json
import shutil
from pycocotools.coco import COCO

def create_test_subset(validation_annotations_path, validation_subset_path, validation_folder_path, test_output_folder, match_threshold=0.9, target_test_count=700):
    
    with open(validation_annotations_path, 'r') as f:
        val_annotations = json.load(f)

    
    val_subset_coco = COCO(validation_subset_path)
    val_subset_image_ids = set(val_subset_coco.getImgIds()) 

    val_coco = COCO(validation_annotations_path)
    val_image_ids = val_coco.getImgIds()
    selected_test_images = []

    for img_id in val_image_ids:
        if img_id in val_subset_image_ids:
            continue  
        if len(selected_test_images) >= target_test_count:
            break 

        ann_ids = val_coco.getAnnIds(imgIds=img_id)
        anns = val_coco.loadAnns(ann_ids)

        
        if len(anns) > 0:
            selected_test_images.append(img_id)


    if not os.path.exists(test_output_folder):
        os.makedirs(test_output_folder)

    
    for img_id in selected_test_images:
        img_info = val_coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        src = os.path.join(validation_folder_path, img_filename)
        dst = os.path.join(test_output_folder, img_filename)
        shutil.copy(src, dst)


    test_annotations_subset = {
        'images': [val_coco.loadImgs(img_id)[0] for img_id in selected_test_images],
        'annotations': [],
        'categories': val_annotations['categories']
    }


    for img_id in selected_test_images:
        ann_ids = val_coco.getAnnIds(imgIds=img_id)
        anns = val_coco.loadAnns(ann_ids)
        test_annotations_subset['annotations'].extend(anns)


    with open(os.path.join(test_output_folder, 'instances_test2017_1_percent.json'), 'w') as f:
        json.dump(test_annotations_subset, f)

    print(f"Test set creation completed. Files saved in {test_output_folder}. Number of images: {len(selected_test_images)}")


# Set the paths
#You will need to update these paths so that they match with where you have the data set stored.
validation_annotations_path = 'C:/coco/annotations_trainval2017/instances_val2017.json'
validation_subset_path = 'C:/coco_subset_20_classes/annotations/instances_val2017.json' 
validation_folder_path = 'C:/coco/val2017'
test_output_folder = 'C:/test2017_1_percent'


create_test_subset(
    validation_annotations_path=validation_annotations_path,
    validation_subset_path=validation_subset_path,
    validation_folder_path=validation_folder_path,
    test_output_folder=test_output_folder,
    match_threshold=0.9,
    target_test_count=700
)
