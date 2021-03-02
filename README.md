# Fashion Mnist

## Data
Neural network implemnentation with pytorch for the fashion mnist dataset.

The dataset can be found at this repo https://github.com/zalandoresearch/fashion-mnist

<img src="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png" width="350" height="350">


## Training
After training for 4 epochs the model got 86% accuracy. On 8 epochs it jumped to 89%.

<img src="/images/train_accuracy.png" width="350" height="350">




## Testing on all data at once
In test_all_data.py it runs all the test data in one batch.
It prints out the accuracy and the confusion matrix.
![confusion matrix](/images/Confusion_matrix.png "Confusion Matrix")




## Testing on a random image
Running test_random_image.py plots a random image with the prediction label next to the true label of the data.
<img src="/images/random_prediction.png" width="350" height="350">


