import torch.nn as nn
import torch.nn.functional as F

#2 conv layers and 3 linear layers
class ConvNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 16, 5, 1, 2)
		self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
		self.fc1 = nn.Linear(7*7*32, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 7*7*32)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)

