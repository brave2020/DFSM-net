
from argparse import ArgumentParser

import os
import numpy as np
import random
import scipy.io as scio
from scipy import stats
import h5py
from PIL import Image
import os,sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision import models
import datetime
import time

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric

# tensorboardX 用于可视化
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: /npip install tensorboardX")



def  rgb2gray(thirdarry):
    # L = R * 299 / 1000 + G * 587 / 1000 + B * 114 / 1000
    R=np.reshape(thirdarry[:,:,[0]],(1024,1024))
    G=np.reshape(thirdarry[:,:,[1]],(1024,1024))
    B=np.reshape(thirdarry[:,:,[2]],(1024,1024))
    NIR=np.reshape(thirdarry[:,:,[3]],(1024,1024))
    gray=R*299/1000+G*587/1000+B*114/1000+NIR*0/1000

    return gray

def meanstd(image):
    bands = np.size(image, 2)
    mean = np.zeros(bands)
    std = np.zeros(bands)
    for i in range(bands):
        image[:,:,i]=image[:,:,i]/255
        # mean[i] = np.mean(image[:,:,i])
        # std[i] = np.var(image[:,:,i])
        # image[:,:,i] = (image[:,:,i] - mean[i])/std[i]

    return image

def matimage_loader(path):
    """
       :param path: .mat image path
       :
       :return: arry image
       """
    matimage = scio.loadmat(path)

    #if "imgMS_HS" in matimage :
        #arryimage = np.float32((matimage['imgMS_HS']))
    if "imgMS_HS" in matimage:
        arryimage=  np.float32((matimage['imgMS_HS']))
        arryimage = meanstd(arryimage)
    else :
        arryimage = np.float32((matimage['Fus_results_values']))
        arryimage = meanstd(arryimage)
    return arryimage

def default_loader(path, channel=3):
    """
    :param path: image path
    :param channel: # image channel
    :return: image
    """
    if channel == 1:
        return Image.open(path).convert('L')   # 转换为灰度图 L = R * 299/1000 + G * 587/1000+ B * 114/1000
    else:
        assert (channel == 3)
        return Image.open(path).convert('RGB')  #



def NonOverlappingCropPatches(im, ref, patch_size=64):
    """
    NonOverlapping Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 32)
    :return: patches
    """
    w=np.size(im,0)
    h=np.size(ref,1)



    patches = ()
    ref_patches = ()
    stride = patch_size
    for i in range(0, h - stride + 1, stride):
        for j in range(0, w - stride + 1, stride):
           # patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch = to_tensor(im[j:j + patch_size,i:i+patch_size])
            patches = patches + (patch,)
            if ref is not None:
                #ref_patch = to_tensor(ref.crop((j, i, j + patch_size, i + patch_size)))
                ref_patch = to_tensor(ref[j:j + patch_size, i:i + patch_size])
                ref_patches = ref_patches + (ref_patch,)

    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)
    else:
        return torch.stack(patches)

def RandomCropPatches(im, ref=None, patch_size=64, n_patches=32):
    """
    Random Crop Patches
    :param im: the distorted image
    :param ref: the reference image if FR-IQA is considered (default: None)
    :param patch_size: patch size (default: 128)
    :param n_patches: numbers of patches (default: 32)
    :return: patches
    """
    w = np.size(im, 0)
    h = np.size(ref, 1)

    patches = ()
    ref_patches = ()
    for i in range(n_patches):
        w1 = np.random.randint(low=0, high=w-patch_size+1) #返回[low,high)内的一个随机数
        h1 = np.random.randint(low=0, high=h-patch_size+1)
        #patch = to_tensor(im.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
        #patch = to_tensor(im[w1:w1 + patch_size, h1:h1 + patch_size,:])#从输入图像中随机裁剪大小为patch_size的子图像块 to_tensor可以理解为归一化处理
        patch = to_tensor(im[w1:w1 + patch_size, h1:h1 + patch_size])
        patches = patches + (patch,) #储存堆叠子图像矩阵
        if ref is not None:
            #ref_patch = to_tensor(ref.crop((w1, h1, w1 + patch_size, h1 + patch_size)))
            #ref_patch = to_tensor(ref[w1:w1 + patch_size, h1:h1 + patch_size,:])
            ref_patch = to_tensor(ref[w1:w1 + patch_size, h1:h1 + patch_size])
            ref_patches = ref_patches + (ref_patch,)

    if ref is not None:
        return torch.stack(patches), torch.stack(ref_patches)     #stack 拼接
    else:
        return torch.stack(patches)


class IQADatast_spe(Dataset):

    def __init__(self, args, status='train', loader=matimage_loader):

        self.status = status
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches
        self.loader = loader

        K = args.K_fold
        k = args.k_test

        ind = scio.loadmat(args.Index_dir)
        index = ind['Index']
        index = index[0,:]
        testindex = index[int((k-1)/K * len(index)):int(k/K * len(index))]
        trainindex1 = index[0:int(((k-1)/K) * len (index)) ]
        trainindex2 = index[int(k/K * len(index)):len (index)]
        trainindex  = np.append(trainindex1,trainindex2)



        if 'train' in status:
            self.index = trainindex
            print("# Train Images: {}".format(len(self.index)))
        if 'test' in status:
            self.index = testindex
            print("# Test Images: {}".format(len(self.index)))
        print('Index:')
        print(self.index)

        self.patches = ()
        self.label = []
        self.label_type = []
        self.fus_names = []
        self.hs_hs_names = []

        fus_method = ['1_PCA', '2_IHS', '3_BDSD', '4_GS', '5_ATWT_M2', '6_MTF_GLP_CBD']
        # MOSmat = scio.loadmat(args.MOS_dir)
        # self.scale = MOSmat['MOS'].max()
        # self.MOS = MOSmat['MOS'] / self.scale

        DMOSmat = scio.loadmat(args.DMOS_dir)
        self.scale = DMOSmat['DMOS'].max()
        #self.DMOS = DMOSmat['DMOS']/self.scale
        self.DMOS = DMOSmat['DMOS'] / 100
        self.index2 = self.index - 1

        for idx in range(len(self.index)):
            for ix in range(6):
                self.hs_hs_names.append(os.path.join(args.hs_ms_dir, (str(self.index[idx]) + ('.mat'))))
                self.fus_names.append(os.path.join(args.fus_dir, str(self.index[idx]), (fus_method[ix] + '.mat')))
                self.label.append(self.DMOS[6 * (self.index2[idx]) + ix])
                self.label_type.append((ix+3)/10)

    def __len__(self):
        return len(self.index)*6

    def __getitem__(self, idx):
        hs_ms = self.loader(self.hs_hs_names[idx])
        fus = self.loader(self.fus_names[idx])

        if self.status == 'train':
            patches = RandomCropPatches(fus, hs_ms, self.patch_size, self.n_patches)
            #patches = NonOverlappingCropPatches(hs_ms, fus, self.patch_size)
        else:
            # patches = NonOverlappingCropPatches(fus, hs_ms, self.patch_size)
            patches = RandomCropPatches(fus, hs_ms, self.patch_size, self.n_patches)
        return patches, torch.Tensor([self.label_type[idx], self.label[idx]])


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()


class Extract_features(nn.Module):
    def __init__(self):
        super(Extract_features, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.chns = [64, 128, 256, 512, 512]
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        self.stage1.add_module(str(0), nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        for x in range(1, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        self.weighted_average=0
    def extract(self, x):
        #h = (x - self.mean) / self.std
        h = self.stage1(x)
        h_relu_1 = h
        h = self.stage2(h)
        h_relu_2 = h
        h = self.stage3(h)
        h_relu_3 = h
        h = self.stage4(h)
        h_relu_4 = h
        h = self.stage5(h)
        h_relu_5 = h

        # h = h.view(-1, 512*256) #1*2048 行向量
        return [h_relu_1, h_relu_2, h_relu_3, h_relu_4, h_relu_5]

    def forward(self, data):
        """
        :param data: distorted and reference patches of images
        :return: quality of images/patches
        """
        x, x_ref = data
        x = x.cuda()
        x_ref = x_ref.cuda()
        batch_size = np.size(x, 0)  #样本数量
        n_patches = np.size(x, 1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)

        c1 = 1e-6
        c2 = 1e-6
        self.overall_var_batch = []
        for i in range(batch_size):
            h_spe = self.extract(x[i])
            h_ref_spe = self.extract(x_ref[i])
            for k in range(len(self.chns)):
                x_mean = h_spe[k].mean([2, 3], keepdim=True)
                y_mean = h_ref_spe[k].mean([2, 3], keepdim=True)
                tmp1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)

                x_var = ((h_spe[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
                y_var = ((h_ref_spe[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
                xy_cov = (h_spe[k] * h_ref_spe[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
                tmp2 = (2 * xy_cov + c2) / (x_var + y_var + c2)

                if k == 0:
                    S1 = tmp1
                    S2 = tmp2
                else:
                    S1 = torch.cat((S1, tmp1), 1)
                    S2 = torch.cat((S2, tmp2), 1)
            overall_var = torch.cat((S1, S2), 1)
            overall_var = torch.squeeze(overall_var)
            self.overall_var_batch.append(overall_var)
        return self.overall_var_batch


class IQA_net(nn.Module):
    def __init__(self, weighted_average=True):
        super(IQA_net, self).__init__()
        self.Extract_features = Extract_features()
        self.weighted_average = 0
        self.fc1_q_all = nn.Linear(2*(64+128+256+512+512), 256)
        self.fc2_q_all = nn.Linear(256, 1)
        # self.fc1_w_all = nn.Linear(2*(64+128+256+512+512), 256) #1472
        # self.fc2_w_all = nn.Linear(256, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = 0

    def forward(self, data):
        x, x_ref = data
        batch_size = np.size(x, 0)  #样本数量
        n_patches = np.size(x, 1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)
        overall_var = self.Extract_features(data)
        for i in range(batch_size):
            h = F.relu(self.fc1_q_all(overall_var[i]))
            q[i] = self.fc2_q_all(h)
        return q

class Classification_net(nn.Module):
    def __init__(self):
        super(Classification_net, self).__init__()
        self.Extract_features = Extract_features()
        self.weighted_average = 0
        # self.fc1_q_all = nn.Linear(2*(64+128+256+512+512), 256)
        # self.fc2_q_all = nn.Linear(256, 6)
        self.fc1_w_all = nn.Linear(2*(64+128+256+512+512), 256) #1472
        self.fc2_w_all = nn.Linear(256, 1)
        self.dropout = nn.Dropout()
        self.weighted_average = 0

    def forward(self, data):
        x, x_ref = data
        batch_size = np.size(x, 0)  #样本数量
        n_patches = np.size(x, 1)
        if self.weighted_average:
            q = torch.ones((batch_size, 1), device=x.device)
        else:
            q = torch.ones((batch_size * n_patches, 1), device=x.device)
        overall_var = self.Extract_features(data)
        for i in range(batch_size):
            h = F.relu(self.fc1_w_all(overall_var[i]))
            q[i] = self.fc2_w_all(h)
        return q

class IQALoss(torch.nn.Module):
    def __init__(self):
        super(IQALoss, self).__init__()

    def forward(self, y_pred, y):
        """
        loss function, e.g., l1 loss
        :param y_pred: predicted values
        :param y: y[0] is the ground truth label
        :return: the calculated loss
        """
        loss1 = F.smooth_l1_loss(y_pred, y[:, 1])
        res=[0] * 6
        indices = y[:, 0]
        for k in range(6):
            idx_tmp = [i for i, x in enumerate(indices) if x == (k+3)/10]
            if idx_tmp:
                tmp = y[idx_tmp, :]
                res[k] = torch.mean(tmp[:, 1])
        pred=[0] * 6
        for k in range(6):
            idx_tmp = [i for i, x in enumerate(indices) if x == (k+3)/10]
            if idx_tmp:
                tmp = y_pred[idx_tmp]
                pred[k] = torch.mean(tmp)
        loss2=0
        for i in range(5):
            for j in range(i+1, 6):
                # tmp_loss = abs(abs(res[i]-res[j]) - abs(pred[i]-pred[j]))
                tmp_loss = abs((res[i] - res[j]) - (pred[i] - pred[j]))
                loss2 = loss2 + tmp_loss
        loss = loss1 + 0.067 * loss2 * 4
        return loss

class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE.

    `update` must receive output of the form (y_pred, y).
    """
    def reset(self):
        self._y_pred = []
        self._y      = []
        self._y_std  = []

    def update(self, output):
        y_pred, y = output

        self._y.append(y[0][1].item())
        #self._y_std.append(y[1].item())
        # n = int(y_pred.size(0) / y[0].size(0))  # n=1 if images; n>1 if patches
        # y_pred_im = y_pred.reshape((y[0].size(0), n)).mean(dim=1, keepdim=True)
        self._y_pred.append(y_pred.item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
       # sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()
     #   outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

        return abs(srocc), abs(krocc), abs(plcc), rmse, mae, q#outlier_ratio


def get_data_loaders(args):
    """ Prepare the train-val-test data
    :param args: related arguments
    :return: train_loader, val_loader, test_loader, scale
    """

    # train_dataset = IQADataset_less_memory(args, 'train')
    # train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                            batch_size=args.batch_size,
    #                                            shuffle=True,
    #                                            num_workers=4)  #
    train_dataset=IQADatast_spe(args, 'train')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)  #

    test_dataset = IQADatast_spe(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset)

    scale = test_dataset.scale

    return train_loader, test_loader, scale

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

def train(IQA_model, Classification_model, IQA_optimizer, Classification_optimizer, train_loader, test_loader, start_epoch):
    best_psnr = 0
    global best_criterion
    global best_epoch
    best_criterion = -10
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        IQA_model = IQA_model.train()
        loss_model = IQALoss()
        prefetcher = DataPrefetcher(train_loader)
        data = prefetcher.next()
        cnt = 0
        while data is not None:
            # if cnt == args.steps_per_epoch:
            #     break
            cnt = cnt + 1
            data_img, label_type, lable = data[0], data[1][:, 0].cuda(), data[1][:, 1].cuda()
            loss_label = data[1].cuda()
            # ---------------------
            # Train IQA_net
            # ---------------------
            IQA_optimizer.zero_grad()
            iqa_pred = IQA_model(data_img).cuda()
            iqa_loss = loss_model(iqa_pred, loss_label)
            iqa_loss.backward()
            torch.nn.utils.clip_grad_norm_(IQA_model.parameters(), 5.)
            IQA_optimizer.step()

            # ---------------------
            # Train Classification
            # ---------------------
            # Classification_optimizer.zero_grad()
            # classification_pred = Classification_model(data_img).cuda()
            # classification_loss = F.smooth_l1_loss(classification_pred, label_type)
            # classification_loss.backward()
            # torch.nn.utils.clip_grad_norm_(Classification_model.parameters(), 5.)
            # Classification_optimizer.step()
            data = prefetcher.next()
        evaluator = create_supervised_evaluator(IQA_model,
                                                metrics={'IQA_performance': IQAPerformance()},
                                                device=device)
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        scale = 1
        SROCC, KROCC, PLCC, RMSE, MAE, Q = metrics['IQA_performance']
        print(
            "Testing Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f}"
                .format(epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE))
        print("部分预测值")
        print(Q[:12])
        writer.add_scalar("SROCC/testing", SROCC, epoch)
        writer.add_scalar("KROCC/testing", KROCC, epoch)
        writer.add_scalar("PLCC/testing", PLCC, epoch)
        writer.add_scalar("RMSE/testing", scale * RMSE, epoch)
        # writer.add_scalar("LOSS/testing", scale * engine.state.output, epoch)

        # scheduler.step(engine.state.epoch)

        # if SROCC > best_criterion and engine.state.epoch / args.epochs > 1 / 6:  #
        if SROCC > best_criterion:
            # if engine.state.epoch/args.epochs > 1/6 and engine.state.epoch % int(args.epochs/10) == 0:
            best_criterion = SROCC
            best_epoch = epoch

            try:
                torch.save(IQA_model.module.state_dict(), args.trained_model_file)
                np.save(args.save_result_file_tr, (Q))
            except:
                torch.save(IQA_model.state_dict(), args.trained_model_file)
                np.save(args.save_result_file_tr, (Q))
                # torch.save(model.state_dict(), args.trained_model_file + str(engine.state.epoch))
        else:
            print("第" + str(best_epoch) + "个epoch最佳，" + "SRCC为" + str(best_criterion))
        print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
        # 保存日志
        result_dir = 'F:\\work2_zh\\work2_fenlei\\checkpoints\\'
        train_log_filename = "spectral_train_log.txt"
        train_log_filepath = os.path.join(result_dir, train_log_filename)
        train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [SRCC] {SRCC_str} [KRCC] {KRCC_str}[PLCC] {PLCC_str} [RMSE] {RMSE_str}\n"
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=epoch,
                                                  SRCC_str=" ".join(["{}".format(SROCC)]),
                                                  KRCC_str=" ".join(["{}".format(KROCC)]),
                                                  PLCC_str=" ".join(["{}".format(PLCC)]),
                                                  RMSE_str=" ".join(["{}".format(scale * RMSE)]))
        with open(train_log_filepath, "a") as f:
            f.write(to_write)

def run(args):
    """
    Run the program
    """
    train_loader, test_loader, scale = get_data_loaders(args)
    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    total_iterations = args.epochs
    lr_ratio = 1
    IQA_model = IQA_net()
    Classification_model = Classification_net()
    IQA_model = IQA_model.to(device)
    Classification_model = Classification_model.to(device)
    IQA_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, IQA_model.parameters()), args.lr,
                                     (0.9, args.decay_ratio))
    Classification_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Classification_model.parameters()), args.lr,
                                     (0.9, args.decay_ratio))
    train(IQA_model, Classification_model, IQA_optimizer, Classification_optimizer, train_loader, test_loader, 0)


    # #储存代码
    # all_params = IQA_model.parameters()
    # regression_params = []
    # for pname, p in IQA_model.named_parameters():
    #     if pname.find('fc') >= 0:
    #         regression_params.append(p)
    # regression_params_id = list(map(id, regression_params))
    # features_params = list(filter(lambda p: id(p) not in regression_params_id, all_params))
    # IQA_optimizer = Adam([{'params': regression_params},
    #                   {'params': features_params, 'lr': args.lr * lr_ratio}],
    #                  lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.StepLR(IQA_optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    #
    # lr_ratio = 1  # feature lr / regression lr
    # if args.model == 'Spectral-FR-new' or args.model == 'DIQaM-FR':
    #     #model = FRnet(weighted_average=args.weighted_average)
    #     model = DISTS(args)
    #     if args.resume is not None:   #最新检查点的路径
    #         model.load_state_dict(torch.load(args.resume))
    # else:
    #     print('Wrong model name!')
    #
    # writer = SummaryWriter(log_dir=args.log_dir)
    # model = model.to(device)
    # print(model)
    #
    # if args.multi_gpu and torch.cuda.device_count() > 1:
    #     print("Using multiple GPU")
    # else:
    #     all_params = model.parameters()
    #     regression_params = []
    #     for pname, p in model.named_parameters():
    #         if pname.find('fc') >= 0:
    #             regression_params.append(p)
    #     regression_params_id = list(map(id, regression_params))
    #     features_params = list(filter(lambda p: id(p) not in regression_params_id, all_params))
    #     optimizer = Adam([{'params': regression_params},
    #                       {'params':  features_params, 'lr': args.lr*lr_ratio}],
    #                      lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    #
    # global best_criterion
    #
    # best_criterion = -1  # SROCC >= -1
    # trainer = create_supervised_trainer(model, optimizer, IQALoss(), device=device)
    # evaluator = create_supervised_evaluator(model,
    #                                         metrics={'IQA_performance': IQAPerformance()},
    #                                         device=device)
    #
    # @trainer.on(Events.ITERATION_COMPLETED)
    # def log_training_loss(engine):
    #     writer.add_scalar("training/loss", scale * engine.state.output, engine.state.iteration)
    #
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(engine):
    #     evaluator.run(test_loader)
    #     metrics = evaluator.state.metrics
    #     SROCC, KROCC, PLCC, RMSE, MAE,Q = metrics['IQA_performance']
    #     print(
    #         "Testing Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f}  LOSS: {:.4f} "
    #         .format(engine.state.epoch, SROCC, KROCC, PLCC,  scale *RMSE,  scale *MAE, scale*engine.state.output))
    #     print("部分预测值")
    #     print(Q[:12])
    #     writer.add_scalar("SROCC/testing", SROCC, engine.state.epoch)
    #     writer.add_scalar("KROCC/testing", KROCC, engine.state.epoch)
    #     writer.add_scalar("PLCC/testing", PLCC, engine.state.epoch)
    #     writer.add_scalar("RMSE/testing", scale * RMSE, engine.state.epoch)
    #     writer.add_scalar("LOSS/testing", scale *engine.state.output, engine.state.epoch)
    #
    #     scheduler.step(engine.state.epoch)
    #
    #     global best_criterion
    #     global best_epoch
    #
    #     #if SROCC > best_criterion and engine.state.epoch / args.epochs > 1 / 6:  #
    #     if SROCC > best_criterion:
    #         # if engine.state.epoch/args.epochs > 1/6 and engine.state.epoch % int(args.epochs/10) == 0:
    #         best_criterion = SROCC
    #         best_epoch = engine.state.epoch
    #
    #         try:
    #             torch.save(model.module.state_dict(), args.trained_model_file)
    #             np.save(args.save_result_file_tr, (Q))
    #         except:
    #             torch.save(model.state_dict(), args.trained_model_file)
    #             np.save(args.save_result_file_tr, (Q))
    #             # torch.save(model.state_dict(), args.trained_model_file + str(engine.state.epoch))
    #     else:
    #         print("第"+str(best_epoch)+"个epoch最佳，"+"SRCC为"+str(best_criterion))
    #     print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    #     #保存日志
    #     result_dir = 'F:\\work2_zh\\work2_tezheng\\checkpoints\\'
    #     train_log_filename = "spectral_train_log.txt"
    #     train_log_filepath = os.path.join(result_dir, train_log_filename)
    #     train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [SRCC] {SRCC_str} [KRCC] {KRCC_str}[PLCC] {PLCC_str} [RMSE] {RMSE_str}\n"
    #     to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
    #                                           epoch=engine.state.epoch,
    #                                           SRCC_str=" ".join(["{}".format(SROCC)]),
    #                                           KRCC_str = " ".join(["{}".format(KROCC)]),
    #                                           PLCC_str = " ".join(["{}".format(PLCC)]),
    #                                           RMSE_str = " ".join(["{}".format(scale * RMSE)]))
    #     with open(train_log_filepath, "a") as f:
    #         f.write(to_write)
    #
    #
    # @trainer.on(Events.COMPLETED)
    # def final_testing_results(engine):
    #     global best_epoch
    #     model.load_state_dict(torch.load(args.trained_model_file))
    #     evaluator.run(test_loader)
    #     metrics = evaluator.state.metrics
    #
    #     SROCC, KROCC, PLCC, RMSE, MAE, Q = metrics['IQA_performance']
    #     print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} %"
    #         .format(best_epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE))
    #     np.save(args.save_result_file, (Q))
    #
    # # kick everything off
    # trainer.run(train_loader, max_epochs=args.epochs)
    #
    # writer.close()

def mkdirs(path):
    # if not os.path.exists(path):
    #     os.makedirs(path)
    os.makedirs(path, exist_ok=True)  #创建目录

if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch (Wa)DIQaM-FR/NR')
    parser.add_argument("--seed", type=int, default=12115505)
    # training parameters
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-4)') #3*1e-6
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 4)') #4
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--decay_interval', type=int, default=50,
                        help='learning rate decay interval (default: 100)') #50
    parser.add_argument('--decay_ratio', type=int, default=0.999,
                        help='learning rate decay ratio (default: 0.8)')  #0.5

    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--K_fold', type=int, default=5,
                        help='K-fold cross-validation (default: 5)')
    parser.add_argument('--k_test', type=int, default=1,
                        help='The k-th fold used for test (1:K-fold, default: 5)')  # last 20%

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to disable TensorBoard visualization')
    parser.add_argument("--test_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='flag whether to use multiple GPUs')
    # data info
    parser.add_argument('--database', default='1IKONOS', type=str,
                        help='database name ')
    # model info
    parser.add_argument('--model', default='Spectral-FR-new', type=str,
                        help='model name ')
    parser.add_argument('--pp', default='0', type=str,
                        help=' ')

    parser.add_argument("--fus_dir", type=str,
                        default='D:/2-DataSet/PAN_MS_fuse_image/mat_datas/4FRtoIQA/1IKONOS/',
                        help="fus image path.")

    parser.add_argument("--hs_ms_dir", type=str,
                        default='D:/2-DataSet/DOMS_data/1Satellite_sensor_data/Data_set/1IKONOS/MS_upsampling/',
                        help="HS_MS path.")

    parser.add_argument("--DMOS_dir", type=str,
                        default='D:/2-DataSet/DOMS_data/4DMOS/1光谱/1IKONOS.mat',
                        help="DMOS path.")

    parser.add_argument("--Index_dir", type=str,
                        default='D:/2-DataSet/DOMS_data/4DMOS/Index200.mat',
                        help="index path.")
    parser.add_argument('--steps_per_epoch', type=int, default=4,
                        help='')
    args = parser.parse_args()
    # 1IKONOS 200; 2QB 500; 3GF1 410; 4WV4 500; 5WV2 500; 6WV3 160;
    args.patch_size = 512  #
    args.n_patches = 1  #

    args.weighted_average = True

    args.log_dir = '{}/EXP{}-{}-{}-{}-lr={}-bs={}'.format(args.log_dir, args.exp_id, args.k_test, args.database,
                                                          args.model, args.lr, args.batch_size)

    mkdirs('checkpoints')
    args.trained_model_file = 'checkpoints/{}-{}-EXP{}-{}-lr={}-bs={}'.format(args.model, args.database, args.exp_id,
                                                                              args.k_test, args.lr, args.batch_size)
    mkdirs('results')
    args.save_result_file = 'results/{}-{}-EXP{}-{}-lr={}-bs={}-pp={}'.format(args.model, args.database, args.exp_id,
                                                                        args.k_test, args.lr, args.batch_size,args.pp)

    mkdirs('results_tr')
    args.save_result_file_tr = 'results_tr/{}-{}-EXP{}-{}-lr={}-bs={}-pp={}'.format(args.model, args.database, args.exp_id,
                                                                        args.k_test, args.lr, args.batch_size,args.pp)

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    run(args)