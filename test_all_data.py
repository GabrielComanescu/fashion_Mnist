from Model import ConvNet
import torch
from torch.utils.data import DataLoader
import data
from sklearn.metrics import confusion_matrix


model_test = ConvNet()
model_test.load_state_dict(torch.load('models/model.pt'))
if torch.cuda.is_available():
    model_test.cuda()
model_test.eval()

train_data, test_data = data.prepare_data()
test_load_all = DataLoader(test_data, batch_size=len(test_data))

print('Test all data')

#Test all test data at once
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:

        if torch.cuda.is_available():
            X_test = X_test.cuda()
            y_test = y_test.cuda()

        X_test = X_test.view(-1,1,28,28)
        y_val = model_test.forward(X_test)

        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()

        #Print accuracy
        print(f'accuracy: {correct.item()*100/len(test_data)}%')

#Print confusion matrix
arr = confusion_matrix(y_test.cpu().view(-1), predicted.cpu().view(-1))
print(arr)
