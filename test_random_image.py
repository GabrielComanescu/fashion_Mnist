import torch
from Model import ConvNet
import data
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt


categories = {
    '0': 'T-shirt/top',
    '1': 'Trouser',
    '2': 'Pullover',
    '3': 'Dress',
    '4': 'Coat',
    '5': 'Sandal',
    '6': 'Shirt',
    '7': 'Sneaker',
    '8': 'Bag',
    '9': 'Ankle boot'
}

#Load model
model_test = ConvNet()
model_test.load_state_dict(torch.load('models/model.pt'))
if torch.cuda.is_available():
    model_test.cuda()
model_test.eval()

#Get random image
x = random.randint(0, 12000)
train_data, test_data = data.prepare_data()
image, label = test_data[x]

if torch.cuda.is_available():
    image = image.cuda()
    label = label.cuda()

with torch.no_grad():

    prediction = model_test.forward(image.view(-1,1,28,28))

    #Show prediction vs correct label
    plt.imshow(image.cpu().reshape(28,28), cmap="gist_yarg")
    plt.title('prediction: ' + categories[str(torch.argmax(prediction).item())] + ' | correct: ' + categories[str(label.item())])
    plt.show()
