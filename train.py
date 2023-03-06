import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from load_ACDC import ACDC_dataset
from UNet import UNet



# iou计算
def iou_mean(pred, target, n_classes=3):
    # n_classes:the number of classes in your dataset,not including background
    ious = []
    iousSum = 0
    pred = pred.view(-1)
    target = np.array(target.cpu())
    target = torch.from_numpy(target)
    target = target.view(-1)
    # Ignore Iou for background class("0")
    for cls in range(1, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
        return iousSum / (n_classes - 1)





def train_net(net, epochs=50, lr=1e-4):
    # 加载训练集
    traindata, testdata = ACDC_dataset()
    plt_iou = []
    # 优化器用Adam
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    # 定义loss算法
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        epoch_iou = []
        # 开始训练
        net.train()
        for step, (patch, mask) in enumerate(tqdm(traindata)):
            patch, mask = patch.to(device), mask.to(device)
            # mask = torch.unsqueeze(mask, 1)
            # 预测结果
            pred = net(patch)
            loss = criterion(pred, mask.long())
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = torch.argmax(pred, dim=1)
                # intersection = torch.logical_and(mask, y_pred)
                # union = torch.logical_or(mask, y_pred)
                # batch_iou = torch.sum(intersection).type(torch.float) / torch.sum(union).type(torch.float)
                batch_iou = iou_mean(y_pred, mask, 3)
                epoch_iou.append(batch_iou)

        epoch_test_iou = []
        net.eval()
        with torch.no_grad():
            for step, (patch, mask) in enumerate(tqdm(testdata)):
                patch, mask = patch.to(device), mask.to(device)
                # mask = torch.unsqueeze(mask, 1)
                pred = net(patch)
                y_pred = torch.argmax(pred, dim=1)
                # intersection = torch.logical_and(mask, y_pred)
                # union = torch.logical_or(mask, y_pred)
                # batch_test_iou = torch.sum(intersection).type(torch.float) / torch.sum(union).type(torch.float)
                batch_test_iou = iou_mean(y_pred, mask, 3)
                epoch_test_iou.append(batch_test_iou)

        print('epoch:', epoch, 'loss:', round(loss.item(), 4),
              'train_Iou:', round(np.mean(epoch_iou), 4),
              'test_Iou:', round(np.mean(epoch_test_iou), 4)
              )
        plt_iou.append(round(np.mean(epoch_test_iou), 4))
        torch.save(net.state_dict(), './weights/epoch_{},loss_{},train_Iou_{},test_Iou_{}.pth'
                   .format(epoch,
                           round(loss.item(), 4),
                           round(np.mean(epoch_iou), 4),
                           round(np.mean(epoch_test_iou), 4)
                           ))
    plt.plot(range(1, epochs + 1), plt_iou, label='IOU')
    plt.legend()
    plt.savefig('./Iou.png')
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类数为4
    net = UNet(input_channels=1, n_classes=4, bilinear=True)
    net.to(device)
    train_net(net)






















