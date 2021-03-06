import torch
import cv2
import torchvision.transforms as transforms
import argparse

from model import CNNModel

parser = argparse.ArgumentParser()      # Parse input arguments
parser.add_argument('-i', '--input', 
    default='input/test_image.jpg',      #Path of image in test
    help='path to the input image')
args = vars(parser.parse_args())


# the computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# list containing all the class labels
labels = [
    'bad', 'good'
    ]
    
# initialize the model and load the trained weights
model = CNNModel().to(device)
checkpoint = torch.load('outputs/model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# define preprocess transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])  

image = cv2.imread(args['input'])
# convert to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image.to(device))
output_label = torch.topk(outputs, 1)
pred_class = labels[int(output_label.indices)]
print(f"Prediction: {pred_class} camera")
    
