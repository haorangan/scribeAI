import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# Load image
image = cv2.imread("emnisttest.png")
image_original = cv2.imread("emnisttest.png")

# Preprocessing
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
image = cv2.bitwise_not(image)

# Find contours
contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = 200
correct_boxes = []
reader = easyocr.Reader(['en'])

def getCharacter(image_file):
    return reader.readtext(image_file, allowlist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Filter contours and store bounding boxes
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w * h > min_contour_area:
        correct_boxes.append((x-3, y-3, w+6, h+6))  # Append as a tuple

# Non-Maximum Suppression function
def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)

    pick = []
    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[indices[:last]]
        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# Apply NMS on the bounding boxes
if correct_boxes:
    correct_boxes = non_max_suppression(correct_boxes)


def sort_boxes(boxes):
    # First sort by y coordinate
    boxes = sorted(boxes, key=lambda b: b[1])

    sorted_boxes = []
    current_row = []
    previous_y = -1
    row_threshold = 20  # Adjust this threshold for row grouping

    for box in boxes:
        x, y, w, h = box
        if abs(y - previous_y) <= row_threshold or previous_y == -1:
            current_row.append(box)
        else:
            # Sort the current row by x coordinate
            current_row.sort(key=lambda b: b[0])
            sorted_boxes.extend(current_row)
            current_row = [box]  # Start a new row
        previous_y = y

    # Add the last row
    if current_row:
        current_row.sort(key=lambda b: b[0])
        sorted_boxes.extend(current_row)

    return sorted_boxes


# Sort the bounding boxes
correct_boxes = sort_boxes(correct_boxes)

sentence = ""
for (x, y, w, h) in correct_boxes:
    cv2.rectangle(image_original, (x, y), (x+w, y+h), (0, 120, 0), 2)

    box = image[y:y+h, x:x+w]
    box = cv2.resize(box, (100, 100))
    box = cv2.cvtColor(box, cv2.COLOR_GRAY2RGB)
    box = cv2.copyMakeBorder(box, top=200, bottom=200, left=200, right=200, borderType=cv2.BORDER_CONSTANT, value=0)

    plt.imshow(box)
    plt.show()

    result = getCharacter(box)

    if len(result) == 0:
        continue

    (_, text, _) = result[0]
    sentence += text

print(sentence)

cv2.imshow("Processed Image", image_original)
cv2.waitKey(0)
cv2.destroyAllWindows()