import pandas as pd
from models import VGG11
import parameter
import os
import torch
from torchvision import transforms
from PIL import Image

CUDA_DEVICE = parameter.cuda_device
MODEL_PATH_SVM = parameter.model_path_svm
DATA_PATH = parameter.data_path
TEST_PATH = parameter.test_path
SAMPLE = parameter.sample
num_classes = len(os.listdir(DATA_PATH))


classes = os.listdir(DATA_PATH)

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

model = VGG11(num_classes=num_classes).cuda(CUDA_DEVICE)
model= torch.load(MODEL_PATH_SVM)
model.eval()

sample_submission = pd.read_csv(SAMPLE)
submission = sample_submission.copy()
for i, filename in enumerate(sample_submission['file']):
    image = Image.open(TEST_PATH+'/'+filename).convert('RGB')
    image = data_transform(image).unsqueeze(0)
    inputs = image.cuda(CUDA_DEVICE)
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    submission['species'][i] = classes[preds[0]]

submission.to_csv('submission.csv', index=False)