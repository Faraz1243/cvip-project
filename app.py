# import model model.pth
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )
        
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1)
        )
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = F.sigmoid(x)
        
        return x

model = CNN()
# Load the saved model state dictionary
model.load_state_dict(torch.load("model.pth"))
model.eval() 

# Define the function to predict the image
def predict_image(image_path):
    # preprocessing of the image
    img = cv.imread(image_path)
    img = cv.resize(img, (128, 128))
    plt.imshow(img)
    b, g, r = cv.split(img)
    img = cv.merge([r,g,b])

    # normalization
    img = np.array(img)
    img = img / 255.0

    # convert the image to tensor
    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2,0,1)
    img = img.unsqueeze(0)


    # predict the image
    with torch.no_grad():
        output = model(img)
        formated_output = output.cpu().detach().numpy().squeeze()

        # thresholding
        ans = 0
        if formated_output >= 0.5:
            return 1
        
        return ans


print(predict_image("./download.jpeg"))