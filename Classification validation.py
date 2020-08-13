import numpy as np
def Evaluation_index(dataloader, model, device):
    model.eval()
    Labels = []
    Objects = []
    with torch.no_grad():
      for bi, Dictionary in enumerate(dataloader):
        Model_INPUT = Dictionary['Model_INPUT']
        for i in range(len(Model_INPUT)):
          Image_IDS = Model_INPUT[i][0].view(-1, 3, 224, 224).to(device)
          Image_LABEL = Model_INPUT[i][1].to(device).long().squeeze(1)

          Logits_Prob = model(Image_IDS)[0].cpu()
          Object = np.argmax(Logits_Prob, axis = 1)
          Labels.extend(Image_LABEL.cpu().detach().numpy().tolist())
          Objects.extend(Object.cpu().detach().numpy().tolist())
        
        
    return Labels, Objects
##Get output of function##
#Labels, Objects = Evaluation_index(Valid_dataloader,
#                                   Trained_RCNN, device)
##Calculating the accuarcy##
#from sklearn.metrics import accuracy_score
#accuracy_score(Labels, Objects)
