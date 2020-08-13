import selectivesearch
from PIL import Image
import numpy as np
import cv2
class RCNN_Dataset(torch.utils.data.Dataset):
  def __init__(self, data):
    self.data = data
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary = {}
    Real_Image = self.data[index][0]
    Annotation = self.data[index][1]['annotation']
    
    Objects = Annotation['object']
    Original_size = Annotation['size']
    Ground_Truths = []
    Names = []
    for i in range(len(Objects)):
      bndbox = Objects[i]['bndbox']
      xmax = int(bndbox['xmax'])
      xmin = int(bndbox['xmin'])
      ymax = int(bndbox['ymax'])
      ymin = int(bndbox['ymin'])
      ground_truth = (xmax, xmin, ymax, ymin)
      name = Objects[i]['name']
      Ground_Truths.append(ground_truth)
      Names.append(name)

    Image_Array = np.asarray(Real_Image)
    _, Regions = selectivesearch.selective_search(Image_Array, 
                                                  scale = 10, 
                                                  min_size = 1000) 
    Green_box = (125, 255, 51)
    Copy_Image = Image_Array.copy()
    Selective_box = []
    for candidate in Regions:
      left = candidate['rect'][0]
      bottom = candidate['rect'][1]
      right = left + candidate['rect'][2]
      top = bottom + candidate['rect'][3]
      Selective_box.append((right, left, top, bottom))

      Copy_Image = cv2.rectangle(Copy_Image,
                                 (left, top),
                                 (right, bottom),
                                 color = Green_box,
                                 thickness = 2)
    
    Model_INPUT = []
    Detected_box = []
    for i in range(len(Ground_Truths)):
      for bi, CD_box in enumerate(Selective_box):
        if Compute_IoU(CD_box, Ground_Truths[i]) > 0.4:
          Detected_box.append(((CD_box), Names[i]))
          x2, x1, y2, y1 = CD_box
          Real_Array = np.asarray(Real_Image)[y1:y2, x1:x2]
          Cropped_Image = Image.fromarray(Real_Array, 'RGB')
          model_ids = Cropped_Image.resize((224, 224))
          model_label = Encoding(Names[i])
          Model_INPUT.append((torch.Tensor(np.asarray(model_ids)), 
                              torch.Tensor([model_label])))
        else:
          x2, x1, y2, y1 = CD_box
          Real_Array = np.asarray(Real_Image)[y1:y2, x1:x2]
          Cropped_Image = Image.fromarray(Real_Array, 'RGB')
          model_ids = Cropped_Image.resize((224, 224))
          model_label = Encoding('background')
          Model_INPUT.append((torch.Tensor(np.asarray(model_ids)),
                              torch.Tensor([model_label])))
    
    Detection_Image = []
    Detected_Image = Image_Array.copy()
    for i in range(len(Detected_box)):
      left = Detected_box[i][0][1]
      right = Detected_box[i][0][0]
      bottom = Detected_box[i][0][2]
      top = Detected_box[i][0][3]

      Detected_Image = cv2.rectangle(Detected_Image,
                                 (left, top),
                                 (right, bottom),
                                 color = Green_box,
                                 thickness = 2)

    Dictionary['Ground_Truths'] = Ground_Truths
    Dictionary['Names'] = Names
    Dictionary['Original_size'] = Original_size
    Dictionary['Model_INPUT'] = Model_INPUT
    Dictionary['Selective_box'] = Selective_box
    Dictionary['Copy_Image'] = Copy_Image
    Dictionary['Detected_box'] = Detected_box
    Dictionary['Detected_Image'] = Detected_Image
    Dictionary['Full_Array'] = np.asarray(Real_Image)
    return Dictionary

    
def Compute_IoU(CD_box, GT_box):
  X_1 = np.maximum(CD_box[1], GT_box[1])
  X_2 = np.minimum(CD_box[0], GT_box[0])
  Y_1 = np.maximum(CD_box[3], GT_box[3])
  Y_2 = np.minimum(CD_box[2], GT_box[2])

  Intersection = np.maximum(X_2 - X_1, 0) * np.maximum(Y_2 - Y_1, 0)
  CD_area = (CD_box[0] - CD_box[1]) * (CD_box[2] - CD_box[3])
  GT_area = (GT_box[0] - GT_box[1]) * (GT_box[2] - GT_box[3])
  Union = CD_area + GT_area - Intersection
  IoU = Intersection / Union

  return IoU

from torch.utils.data import DataLoader
BATCH_SIZE = 1
Train_dataloader = DataLoader(RCNN_Dataset(Train_data),
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              drop_last = True)
