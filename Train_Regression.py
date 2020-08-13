def Make_Forms(Tuple):
  x_2, x_1, y_2, y_1 = Tuple
  T_x = (x_1 + x_2) * 0.5
  T_y = (y_1 + y_2) * 0.5
  T_w = (x_2 - x_1)
  T_h = (y_2 - y_1)

  return (T_x, T_y, T_w, T_h)
 
def To_Device(Tuple):
  t1, t2, t3, t4 = Tuple
  
  t1 = t1.to(device)
  t2 = t2.to(device)
  t3 = t3.to(device)
  t4 = t4.to(device)
  
  return (t1, t2, t3, t4)

from tqdm import tqdm
def Train_Regression(dataloader, model, optimizer, device):
  model.train()
  BOOK = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0
  for bi, Dictionary in enumerate(BOOK):
    Box_tensor = Dictionary['Box_tensor']
    Ground_Truths = Dictionary['Ground_Truths']
    Reg_INPUT = Dictionary['Reg_INPUT']
    for i in range(len(Box_tensor)):
      Reg_Input = torch.Tensor(Reg_INPUT[i].float()).view(-1, 3, 224, 224).to(device)
      Reg_Label = torch.Tensor(Ground_Truths[i]).to(device)
      
      model.zero_grad()
      Reg_Logits = model(Reg_Input)
      
      d_x, d_y, d_w, d_h = Reg_Logits
      p_x, p_y, p_w, p_h = Make_Forms(Box_tensor[i])
      g_x, g_y, g_w, g_h = Make_Forms(Ground_Truths[i])
      
      d_x, d_y, d_w, d_h = To_Device((d_x, d_y, d_w, d_h))
      p_x, p_y, p_w, p_h = To_Device((p_x, p_y, p_w, p_h))
      g_x, g_y, g_w, g_h = To_Device((g_x, g_y, g_w, g_h))
      
      Predicted_G = (p_w * d_x + p_x, p_h * d_y + p_y, p_w * torch.exp(d_w), p_h * torch.exp(d_h))
      G = g_x, g_y, g_w, g_h 
      
      LOSS = Regression_Loss(Predicted_G, G)
      LOSS1 = torch.sqrt(LOSS[0])
      LOSS2 = torch.sqrt(LOSS[1])
      LOSS3 = torch.sqrt(LOSS[2])
      LOSS4 = torch.sqrt(LOSS[3])
      LOSS1.backward(retain_graph = True)
      LOSS2.backward(retain_graph = True)
      LOSS3.backward(retain_graph = True)
      LOSS4.backward()
      loss = (LOSS[0] + LOSS[1] + LOSS[2] + LOSS[3]) * 0.25
      
      optimizer.step()
      optimizer.zero_grad()
      
      total_loss += loss.item()
  Average_Train_Loss = total_loss / len(dataloader)
  print(" Average Train Loss: {0:.2f}".format(Average_Train_Loss))
  
def FIT_Regressions(dataloader, model, Epochs, Learning_rate):
  optimizer = torch.optim.AdamW(model.parameters(), lr = Learning_rate)

  for i in range(Epochs):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    Train_Regression(dataloader, model, optimizer, device)
  torch.save(model, '/content/gdrive/My Drive/' + f'Regression_MODEL')

FIT_Regressions(Regression_dataloader, Regression_MODEL, Epochs = 7 , Learning_rate = 0.002)
