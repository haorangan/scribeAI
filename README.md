ScribeAI is a project that turns handwritten text from user-uploaded images into digital text. The heart of the system is an algorithm that classifies letters using thresholding, noise reduction, and bounding box detection with OpenCV. After isolating each letter, we pass the bounding boxes to a CNN trained on the EMNIST dataset for classification.

Challenges
While developing ScribeAI, I ran into some challenges, especially with the accuracy of letter recognition. The EMNIST dataset, which is great for single letters, didn’t quite cut it for full words, leading to some misclassifications—especially with messy handwriting or overlapping letters.

Future Improvements

Diverse Training Data: Training on a larger, more varied dataset with complete words and different handwriting styles would make the model more reliable. EMNIST only trains the model on a specific type of image.
RNN Integration: Adding RNNs could help the model understand letter sequences better, improving word recognition.
Data Augmentation: This could expose the model to a wider range of handwriting styles during training.
