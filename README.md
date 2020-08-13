# RCNN_For_Image_Detection

RCNN
 0. Set BASE
 1. Load VOC Detection data
  1.1 Data minimizing
  1.2 Encoding words
  

 2. Dataset(include selective_searching & Warping)

 3. RCNN_Model

 4. RCNN_Loss

 5. Train(RCNN FINE TUNING)
  
 6. Train SVM(For each 21 labels)
  6.1 But it wastes too many times, so replacing it by softmax(softmax is in Crossentropy)
  6.2 Classification validation
 
 7. Get Detection box before bounding-box regression by Dataset Module
 
 8. bounding_box regressor
   predict a new bounding box using a class-specific bounding-box regressor
   here we regress from features computed by the CNN
   Train pair = {(P, G)}

   P = (P_x, P_y, P_w, P_h) pixel coordinates of the center amd width and height
   G = (G_x, G_y, G_w, G_h) in the same way

   Goal is to learn a transformation that maps a 'proposed box P to G'

   Parameterize the transformation in terms of 4 functions
   - d_x(P), d_y(P), d_w(P), d_h(P)
    First two specify a scale-invariant translation of the P_center
    Second two specify log-space translations of width, height of P

   Each function d_?(P) is modeled as a linear function of the pool_5 features of P
   here the pool_5 is same with Model_OUTPUT[1]

   regression target t_* for the training pair (P, G) are defined as 
    t_x = (G_x - P_x) / P_w
    t_y = (G_y - P_y) / P_w
    t_w = log(G_w / P_w)
    t_h = log(G_h / P_h)
