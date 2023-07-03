import cv2
import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
#Tra ve hinh anh va nhan cua anh do
import data

#Load Image
label = 9
count = 10
inputs, result, count = data.loadata(label, count)

#result = [1,0,0,0,0,0,0,0,0,0]

#Chuyen ve kieu du lieu cua pytorch
inputs = torch.tensor(inputs, dtype=torch.float32)
result = torch.tensor(result, dtype=torch.float32)

print(inputs)
print(inputs.shape) #torch.Size([2352])

#modell
model = nn.Sequential(
                nn.Linear(2352,100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100,10),
                nn.Sigmoid()
)

print(model) 
#Sequential(
#  (0): Linear(in_features=2352, out_features=100, bias=True)
#  (1): ReLU()
#  (2): Linear(in_features=100, out_features=100, bias=True)
#  (3): ReLU()
#  (4): Linear(in_features=100, out_features=10, bias=True)
#  (5): Sigmoid()
#)

loss_fn = nn.BCELoss() #binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Qua trinh tao tao 
n_epochs = 1 #Chuyển toàn bộ tập dữ liệu huấn luyện cho mô hình một lần 
batch_size = 1 # Một hoặc nhiều mẫu được chuyển đến mô hình, từ đó thuật toán giảm dần độ dốc sẽ được thực thi cho một lần lặp 

#truyen inputs vao cho model
#y_pred = model(inputs)

#print(y_pred)
#tensor([0.5178, 0.5283, 0.4611, 0.5160, 0.5176, 0.4748, 0.4862, 0#.5328, 0.5041, 0.4784], grad_fn=<SigmoidBackward0>)


for epoch in range(n_epochs):
    for i in range(batch_size):
        Xbatch = inputs
        y_pred = model(inputs)
        ybatch = result
        loss = loss_fn(y_pred,ybatch)
        optimizer.zero_grad()
        #Lan chuyen nguoc 
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

torch.save(model, '/home/tuankhanh/Desktop/neural-networks/save/core.pt')

