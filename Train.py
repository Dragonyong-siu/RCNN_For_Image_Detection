from tqdm import tqdm
def Train_Epoch(dataloader, model, optimizer, device):
  model.train()
  Book = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0
  for bi, Dictionary in enumerate(Book):
    Model_INPUT = Dictionary['Model_INPUT']
    for i in range(len(Model_INPUT)):
      Image_IDS = Model_INPUT[i][0].view(-1, 3, 224, 224).to(device)
      Image_LABEL = Model_INPUT[i][1].to(device).long().squeeze(1)

      model.zero_grad()

      Logits = model(Image_IDS)

      Loss = RCNN_Loss(Logits[0], Image_LABEL)
      Loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      total_loss += Loss.item()
  Average_Train_Loss = total_loss / len(dataloader)
  print(" Average Train Loss: {0:.2f}".format(Average_Train_Loss))

def FIT_Epochs(dataloader, model, Epochs, Learning_rate):
  optimizer = torch.optim.AdamW(RCNN_Model.parameters(), lr = Learning_rate)

  for i in range(Epochs):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    Train_Epoch(dataloader, model, optimizer, device)
  torch.save(RCNN_Model, '/content/gdrive/My Drive/' + f'RCNN_Model')

FIT_Epochs(Train_dataloader, RCNN_Model, Epochs = 3, Learning_rate = 0.0002)
