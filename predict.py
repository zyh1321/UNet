import torch
import numpy as np
from matplotlib import pyplot as plt
from load_ACDC import ACDC_dataset
from UNet import UNet
import pylab
from train import iou_mean


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

traindata, testdata = ACDC_dataset()


Iou = []

def Predict_result():
    global model, img, label, state_dict, elbo, intersection, union
    model = UNet(input_channels=1, n_classes=4, bilinear=True)
    model.to('cuda')
    img, label = next(iter(testdata))
    img = img.to('cuda')
    label = label.to('cuda')
    # label = torch.unsqueeze(label, 1)
    state_dict = torch.load('./weights/epoch_48,loss_0.0041,train_Iou_0.4884,test_Iou_0.4264.pth')
    model.load_state_dict(state_dict)
    model.eval()
    # torch.Size([batch_size, 2, 128, 128])
    pred = model(img)
    pred = torch.argmax(pred, dim=1)
    # pred = torch.unsqueeze(pred, dim=1)
    for i in range(pred.shape[0]):
        if np.max(label[i].cpu().numpy()) > 0 and np.max(pred[i].cpu().numpy()) > 0:
            # intersection = torch.logical_and(torch.squeeze(label), torch.argmax(pred, dim=1))

            # intersection = torch.logical_and(label[i], pred[i])
            # union = torch.logical_or(label[i], pred[i])
            # batch_iou = torch.sum(intersection) / torch.sum(union)
            batch_iou = iou_mean(pred, label, 3)
            Iou.append(batch_iou)

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 3, 1)
        plt.imshow(img[i].permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(label[i].cpu().numpy())
        plt.subplot(1, 3, 3)
        plt.imshow(pred[i].cpu().detach().numpy())
        pylab.show()



if __name__ == '__main__':
    Predict_result()
    print(round(np.mean(Iou), 4))
