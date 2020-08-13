import torchvision
from torchvision import transforms
Train_data = torchvision.datasets.VOCDetection(root = './data',
                                  year = '2012',
                                  image_set = 'train',
                                  download = True,
                                  transform = None,
                                  target_transform = None,
                                  transforms = None)

# Minimizing data
train_data = []
for i in range(0, 5000, 25):
  train_data.append(Train_data[i])
Train_data = train_data
