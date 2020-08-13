class Regression_Model(nn.Module):
  def __init__(self):
    super(Regression_Model, self).__init__()
    self.Make_dimension_1 = nn.Linear(4096, 4096)
    self.Make_dimension_2 = nn.Linear(4096, 4)
    
  def forward(self, Image_array):
    Model_OUTPUT = Trained_RCNN(Image_array)[1]
    Model_OUTPUT = self.Make_dimension_1(Model_OUTPUT)
    D = self.Make_dimension_2(Model_OUTPUT)
    d_x, d_y, d_w, d_h = D[0][0], D[0][1], D[0][2], D[0][3]
    return d_x, d_y, d_w, d_h

MODEL = Regression_Model()
Regression_MODEL = MODEL.to(device)
