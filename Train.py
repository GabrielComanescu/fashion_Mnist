import numpy as np
import data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from Model import ConvNet
import time

if __name__ == '__main__':

	#GET DATA
	train_loader, test_loader = data.get_data()

	#DEFINE MODEL
	model = ConvNet()

	#CRITERION AND OPTIMIZER
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

	if torch.cuda.is_available():
		model.cuda()
		criterion.cuda()

	start_time = time.time()
	epochs = 8		#5 - 6 epochs got me to 88% accuracy. After more than 8 epochs it starts overfitting
	train_losses = []
	test_losses = []
	train_correct = []
	test_correct = []



	#TRAIN
	for i in range(epochs):
		trn_corr = 0
		tst_corr = 0
		model.train()

		for b, (X_train, y_train) in enumerate(train_loader):
			b += 1

			if torch.cuda.is_available():
				X_train = X_train.cuda()
				y_train = y_train.cuda()

			#Data reshape for conv layers
			X_train = X_train.view(-1,1,28,28)

			optimizer.zero_grad()

			y_pred = model.forward(X_train)
			loss = criterion(y_pred, y_train)

			#Track correct guesses
			predicted = torch.max(y_pred.data, 1)[1]
			batch_corr = (predicted == y_train).sum()
			trn_corr += batch_corr

			loss.backward()
			optimizer.step()

			#Print loss and acurracy every few epochs
			if b%48 == 0:
				print(f'epoch: {i}  batch: {b}  loss: {loss.item()}   accuracy: {trn_corr.item()*100/(100*b)}')

		train_losses.append(loss)
		train_correct.append(trn_corr)
		model.eval()

		with torch.no_grad():

			for b, (X_test, y_test) in  enumerate(test_loader):
				b+=1

				if torch.cuda.is_available():
					X_test = X_test.cuda()
					y_test = y_test.cuda()

				X_test = X_test.view(-1,1,28,28)
				y_val = model.forward(X_test)

				predicted = torch.max(y_val.data, 1)[1] 
				tst_corr += (predicted == y_test).sum()

		loss = criterion(y_val, y_test)
		test_losses.append(loss)
		test_correct.append(tst_corr)
		print(f'test  accuracy: {tst_corr.item()*100/(100*b)}')


print(f'{time.time() - start_time}  seconds')

torch.save(model.state_dict(), 'models/model.pt')

plt.plot([t/480 for t in train_correct], label='training accuracy')
plt.plot([t/120 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()
