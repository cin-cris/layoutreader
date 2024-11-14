import json
import cv2

# Load JSON file
json_path = './aur-684_multimodal_document_parsing_v2/label/P69_84_8_16.json'
image_path = './aur-684_multimodal_document_parsing_v2/input/P69_84_8_16.png'

with open(json_path, 'r') as f:
    data = json.load(f)

# Load the image
image = cv2.imread(image_path)

# Set font parameters for labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_color = (0, 0, 0)  # Red color for text
font_thickness = 1

# Draw bounding boxes with labels
for region in data['regions']:
    shape_attributes = region['shape_attributes']
    alias_id = region['region_attributes'].get('alias_id', "-1")
    
    x = shape_attributes['x']
    y = shape_attributes['y']
    width = shape_attributes['width']
    height = shape_attributes['height']
    
    # Draw red rectangle on the image
    top_left = (x, y)
    bottom_right = (x + width, y + height)
    cv2.rectangle(image, top_left, bottom_right, color=(0, 0, 255), thickness=2)
    
    # Put label above the bounding box
    label_position = (x, y - 10) if y - 10 > 10 else (x, y + 20)
    cv2.putText(image, str(alias_id), label_position, font, font_scale, font_color, font_thickness)


# Save or display the output image with bounding boxes
output_image_path = './test_with_bounding_boxes.png'
cv2.imwrite(output_image_path, image)

print(f"Bounding boxes drawn. Image saved at {output_image_path}")
