import torch
from transformers import DetrForObjectDetection
from pycocotools.coco import COCO
from PIL import Image
from torchvision import transforms
import json
from pycocotools.cocoeval import COCOeval

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True)

# Load the checkpoint file
checkpoint = torch.load("/content/drive/MyDrive/detr/second_training/checkpoint.pth", map_location=device)

# Extract only the model's weights from the checkpoint
model_weights = checkpoint["model"] if "model" in checkpoint else checkpoint  # Adjust in case of direct state_dict

# Load the weights into the model
model.load_state_dict(model_weights, strict=False)  # `strict=False` allows loading with missing or extra keys
model.eval()  # Set to evaluation mode

# Move model to GPU if available
model.to(device)


# Path to the test annotations file
test_annotations_path = "/content/drive/MyDrive/test2017/annotations/instances_test2017.json"  # Replace with your actual path

# Initialize COCO API for test set
test_coco = COCO(test_annotations_path)

# Get the image IDs for the test dataset
test_image_ids = test_coco.getImgIds()



# Define the transform for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Adjust size if necessary based on your model's input requirements
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize an empty list to store the predictions in COCO format
coco_predictions = []

# Loop through each image in the test subset
for img_id in test_image_ids:
    img_info = test_coco.loadImgs(img_id)[0]
    img_path = f"/content/drive/MyDrive/test2017/test2017/{img_info['file_name']}"  # Replace with actual path to test dataset
    image = Image.open(img_path).convert("RGB")
    width, height = image.size  # Get original image dimensions

    # Preprocess the image
    input_image = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_image)

    # Extract predictions
    logits = outputs.logits[0]  # Class scores
    boxes = outputs.pred_boxes[0].cpu()  # Bounding boxes in normalized format

    # Process predictions
    probs = logits.softmax(-1)[:, :-1]  # Exclude the "no-object" class
    scores, labels = probs.max(-1)

    # Define a confidence threshold
    confidence_threshold = 0.5
    for score, label, box in zip(scores, labels, boxes):
        if score >= confidence_threshold:
            # Convert normalized [cx, cy, width, height] to [x_min, y_min, width, height]
            cx, cy, w, h = box.tolist()
            xmin = (cx - w / 2) * width
            ymin = (cy - h / 2) * height
            abs_width = w * width
            abs_height = h * height

            # Ensure the bounding box dimensions are valid
            if abs_width > 0 and abs_height > 0:
                coco_predictions.append({
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": [xmin, ymin, abs_width, abs_height],
                    "score": score.item()
                })

print(f"Completed inference on {len(test_image_ids)} images.")

# Example to save the predictions to a JSON file in COCO format
predictions_path = '/content/drive/MyDrive/test2017_predictions.json'  # Set your desired output path
with open(predictions_path, 'w') as f:
    json.dump(coco_predictions, f, indent=4)

print(f"Predictions saved to {predictions_path}")

# Load the ground truth annotations and your model's predictions
coco_gt = test_coco  # Ground truth
coco_dt = coco_gt.loadRes(predictions_path)  # Load the saved predictions

# Initialize the COCOeval object
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

# Run the evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()