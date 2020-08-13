# Get Detection box before bounding-box regression by Dataset Module
import matplotlib.pyplot as plt
class Regression_Dataset(torch.utils.data.Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary = {}
    Full_Array = RCNN_Dataset(self.data)[index]['Full_Array']
    Ground_Truths = RCNN_Dataset(self.data)[index]['Ground_Truths']
    Detected_box = RCNN_Dataset(self.data)[index]['Detected_box']
    Model_INPUT = RCNN_Dataset(self.data)[index]['Model_INPUT']
    Image_Array = np.asarray(self.data[index][0])
    if len(Detected_box) == 0:
      Detected_box = RCNN_Dataset(self.data)[index - 1]['Detected_box']
    IOUs = []
    for i in range(len(Ground_Truths)):
      for j in range(len(Detected_box)):
        iou = Compute_IoU(Detected_box[j][0], Ground_Truths[i])
        IOUs.append((iou, j))

    BEST_Num = []
    for k in range(len(Ground_Truths)):
      Part = IOUs[:len(Detected_box)]
      Green_num = max(Part)[1]
      BEST_candidate = Part[Green_num]
      BEST_Num.append(BEST_candidate[1])
      del IOUs[:len(Detected_box)]

    Box_tensor = []
    for i in range(len(BEST_Num)):
      Box_tensor.append(Detected_box[BEST_Num[i]][0])

    Green_box = (125, 255, 51)
    for i in range(len(Box_tensor)):
      left = Box_tensor[i][1]
      right = Box_tensor[i][0]
      bottom = Box_tensor[i][2]
      top = Box_tensor[i][3]

      Image_Array = cv2.rectangle(Image_Array,
                                 (left, top),
                                 (right, bottom),
                                 color = Green_box,
                                 thickness = 2)
    Reg_INPUT = []
    for i in range(len(Box_tensor)):
      x2, x1, y2, y1 = Box_tensor[i]
      Crop_Image = Image.fromarray(Full_Array[y1:y2, x1:x2],
                                   'RGB')
      reg_input = Crop_Image.resize((224, 224))
      Reg_INPUT.append(np.asarray(reg_input))
      
    Dictionary['Image_Array'] = Image_Array
    Dictionary['Box_tensor'] = Box_tensor
    Dictionary['Ground_Truths'] = Ground_Truths
    Dictionary['Reg_INPUT'] = Reg_INPUT
    Dictionary['Full_Array'] = Full_Array
    return Dictionary

BATCH_SIZE = 1
Regression_dataloader = DataLoader(Regression_Dataset(Train_data),
                                   batch_size = BATCH_SIZE,
                                   shuffle = True,
                                   drop_last = True)
