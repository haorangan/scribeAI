import cv2
import torch
from EMNIST import EMNISTNetwork, test_data
import matplotlib

def classify_character(cropped_character):
    model = EMNISTNetwork()
    model.load_state_dict(torch.load(f="./models/EMNISTmodel.pt", weights_only=True))

    model.eval()
    with torch.inference_mode():
        # Convert to tensor and process
        tensor_image = torch.tensor(cropped_character, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 28, 28]

        tensor_image = tensor_image.unsqueeze(0)

        tensor_image = tensor_image / 255.0


        y_pred = model(tensor_image)
        y_pred = torch.softmax(y_pred, dim=1)
        index = torch.argmax(y_pred, dim=1).item()
        return index

# Load and preprocess the image
image = cv2.imread("images/words.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

inverted = cv2.bitwise_not(gray)

_, binary = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)

blurred = cv2.GaussianBlur(binary, (5, 5), 0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

# Detect contours
contours, _ = cv2.findContours(opened, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Filter and process each letter
recognized_text = []
min_contour_area = 100
for contour in contours:
    if cv2.contourArea(contour) > min_contour_area:
        # Get the bounding box for the individual letter
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the individual letter
        cropped_letter = binary[y:y + h, x:x + w]

        # Resize and process the letter
        resized_letter = cv2.resize(cropped_letter, (28, 28))
        letter = classify_character(resized_letter)
        recognized_text.append(letter)

# Convert class indices back to characters
classes = test_data.classes
final_output = ''.join([classes[index] for index in recognized_text])

print("Recognized Text:", final_output)

image_with_boxes = image.copy()
for contour in contours:
    if cv2.contourArea(contour) > min_contour_area:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("bounding", opened)
cv2.waitKey(0)
cv2.destroyAllWindows()