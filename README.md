ScribeAI is a small project I decided to work on that turns handwritten text from user-uploaded images into digital text. The core of the project is an algorithm that classifies letters using thresholding, noise reduction, and bounding box detection with OpenCV. After isolating each letter, I pass the bounding boxes to a CNN trained on the EMNIST dataset for classification.

*A second version found in the ignore folder uses a pre-trained model from EasyOCR as well as NMS for better bounding box detection.

Challenges

While developing ScribeAI, I ran into some challenges, especially with the accuracy of letter recognition. The EMNIST dataset, which is great for single letters, didn’t quite cut it for full words, leading to some misclassifications—especially with messy handwriting or overlapping letters. The bounding boxes were also difficult to get right, especially when they overlapped or when letters were too close together to distinguish.

Future Improvements

Diverse Training Data: Training on a larger, more varied dataset with complete words and different handwriting styles would make the model more reliable. EMNIST only trains the model on a specific type of image.

RNN Integration: Adding RNNs could help the model understand letter sequences better, improving word recognition.

Data Augmentation: This could expose the model to a wider range of handwriting styles during training.

![image](https://github.com/user-attachments/assets/bd076edb-9162-4593-8cd4-b0256c0465aa)
