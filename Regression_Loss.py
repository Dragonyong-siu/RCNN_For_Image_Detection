def Regression_Loss(Predicted_G, G):
  (predG_x, predG_y, predG_w, predG_h) = Predicted_G
  G_x, G_y, G_w, G_h = G
  Loss_Function = nn.MSELoss()
  Loss_x = Loss_Function(torch.Tensor(predG_x.cpu()), torch.Tensor(G_x.cpu()))  
  Loss_y = Loss_Function(torch.Tensor(predG_y.cpu()), torch.Tensor(G_y.cpu())) 
  Loss_w = Loss_Function(torch.Tensor(predG_w.cpu()), torch.Tensor(G_w.float().cpu())) 
  Loss_h = Loss_Function(torch.Tensor(predG_h.cpu()), torch.Tensor(G_h.float().cpu())) 
  return torch.sqrt(Loss_x), torch.sqrt(Loss_y), torch.sqrt(Loss_w), torch.sqrt(Loss_h)
