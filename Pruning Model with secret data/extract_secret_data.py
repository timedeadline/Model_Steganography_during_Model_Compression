import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
import random


class VGG(nn.Module):
    def __init__(self,features,num_classes=10,init_weights=False):
        super(VGG,self).__init__()
        self.features = models.__dict__[features](pretrained=False).features
        self.classifier = nn.Sequential(  # 分类部分的网络
            nn.Linear(512,512),
            nn.Linear(512, num_classes),
        )
        # add the quantize part
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        x = self.quant(x)
        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias,0)
            elif isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(module.bias,0)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy_origin(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy_origin(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5


def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed / num_images * 1000
    # return elapsed

def load_model(model_file):
    model_name = "vgg16_bn"
    model = VGG(features=model_name,num_classes=10,init_weights=False)
    check_point = torch.load(model_file)
    model.load_state_dict(check_point['state_dict'])
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    size = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size


def prune_network(args, network=None):
    device = torch.device("cuda" if args.gpu_no >= 0 else "cpu")
    args.vgg = 'vgg16_bn'
    args.data_set = 'CIFAR10'
    args.load_path = "./trained_models/check_point.pth"
    args.prune_layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9", "conv10", "conv11", "conv12", "conv13"]
    args.prune_channels = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    args.independent_prune_flag = False
    if network is None:
        network = VGG(args.vgg, args.data_set)
        if args.load_path:
            check_point = torch.load(args.load_path)
            network.load_state_dict(check_point['state_dict'])

    # prune network
    network = prune_step(network, args.prune_layers, args.prune_channels, args.independent_prune_flag)
    network = network.to(device)

    return network


def prune_step(network, prune_layers, prune_channels, independent_prune_flag):
    network = network.cpu()

    prune_index = 0  # count for indexing 'prune_channels'
    conv_count = 1  # conv count for 'indexing_prune_layers'
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None  # residue is need to prune by 'independent strategy'
    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            if dim == 1:
                new_, residue = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                # 异或操作，相同为0，不同为1
                dim ^= 1
            if 'conv%d' % conv_count in prune_layers:
                channel_index = get_channel_index(network.features[i].weight.data, prune_channels[prune_index], residue)
                new_ = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1
                prune_index += 1
            else:
                residue = None
            conv_count += 1

        elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(network.features[i], channel_index)
            network.features[i] = new_

    # update to check last conv layer pruned
    if 'conv13' in prune_layers:
        network.classifier[0] = get_new_linear(network.classifier[0], channel_index)

    return network


def get_channel_index(kernel, num_elimination, residue=None):
    # get cadidate channel index for pruning
    ## 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)

    vals, args = torch.sort(sum_of_kernel)

    return args[:num_elimination].tolist()


def index_remove(tensor, dim, index, removed=False):
    if tensor.is_cuda:
        tensor = tensor.cpu()

    select_index = list(set(range(tensor.size(dim))) - set(index))
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor


def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)
        new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation)

        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        new_conv.bias.data = conv.bias.data

        return new_conv, residue


def get_new_norm(norm, channel_index):
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)

    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)

    return new_norm


def get_new_linear(linear, channel_index):
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)
    new_linear.bias.data = linear.bias.data

    return new_linear


class Parser():
    def __init__(self, gpu_no=0, train_flag=False, resume_flag=False, prune_flag=False, retrain_flag=False,
                 retrain_epoch=20, retrain_lr=0.001, data_set='CIFAR10', data_path='../', vgg='vgg16_bn', start_epoch=0, epoch=350,
                 batch_size=128, num_workers=0, lr=0.1, lr_milestone=[150, 250], lr_gamma=0.1, momentum=0.9, weight_decay=5e-4,
                 imsize=None, cropsize=32, crop_padding=4, hflip=0.5, print_freq=100, load_path=None,
                 save_path='./trained_models/', independent_prune_flag=False, prune_layers=None, prune_channels=None, hiding_flag=False):
        # cpu: -1, gpu: 0 ~ n
        self.gpu_no = gpu_no
        # flag for training network 训练网络的flag，是否训练
        self.train_flag = train_flag
        # flag for resume training  是否进行恢复性重新训练
        self.resume_flag = resume_flag
        # flag for pruning network 是否网络剪枝
        self.prune_flag = prune_flag
        # flag for retraining pruned network 是否重训练剪枝网络
        self.retrain_flag = retrain_flag
        # number of epoch for retraining pruned network 重新训练修剪网络的历元数
        self.retrain_epoch = retrain_epoch
        # learning rate for retraining pruned network 再训练修剪网络的学习率
        self.retrain_lr = retrain_lr
        # Data set for training network 训练网络的数据集
        self.data_set = data_set
        # Path of dataset 数据集路径
        self.data_path = data_path
        # version of vgg network VGG网络版本
        self.vgg = vgg
        # start epoch for training network 开始训练网络的epoch
        self.start_epoch = start_epoch
        # number of epoch for training network 训练网络的历元数
        self.epoch = epoch
        # batch size 批量大小
        self.batch_size = batch_size
        # number of workers for data loader 数据加载器的工作程序数
        self.num_workers = num_workers
        # learning rate 学习率
        self.lr = lr
        # list of epoch for adjust learning rate 用于调整学习率的历元列表
        self.lr_milestone = lr_milestone
        # factor for decay learning rate 衰减学习率因子
        self.lr_gamma = lr_gamma
        # momentum for optimizer 优化器的动量
        self.momentum = momentum
        # factor for weight decay in optimizer 优化器中的权重衰减因子
        self.weight_decay = weight_decay
        # size for image resize 调整图像大小的大小
        self.imsize = imsize
        # size for image crop 图像裁剪尺寸
        self.cropsize = cropsize
        # size for padding in image crop 在图像裁剪中填充的大小
        self.crop_padding =crop_padding
        # probability of random horizontal flip 随机水平翻转的概率
        self.hflip = hflip
        # print frequency during training 训练期间打印频率
        self.print_freq = print_freq
        # trained model load path to prune 训练好的模型加载路径进行修剪
        self.load_path = load_path
        # model save path 模型保存路径
        self.save_path = save_path
        # prune multiple layers by "independent strategy" 采用“独立策略”进行多层剪枝
        self.independent_prune_flag = independent_prune_flag
        # layer index for pruning 修剪层索引
        self.prune_layers = prune_layers
        # number of channel to prune layers 修剪层的通道数
        self.prune_channels = prune_channels
        # 是否嵌入信息
        self.hiding_flag = hiding_flag


args = Parser()
args.gpu_no = 0

args.data_set = 'CIFAR10'
args.vgg = 'vgg16_bn'
args.save_path = './trained_models/'


def get_parameter():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)

    print("-*-" * 10 + "\n\tArguments\n" + "-*-" * 10)
    for key, value in vars(args).items():
        print("%s: %s" % (key, value))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("Make dir: ", args.save_path)

    torch.save(args, args.save_path + "arguments.pth")

    return args


args = get_parameter()


# 加载原始模型

# origin_net_data = torch.load("jianzhi_stego_0bits.pth")
# saved_model_dir = 'trained_models/'
# float_model_file = 'check_point.pth'
# origin_net = load_model(saved_model_dir + float_model_file)
# origin_net = prune_network(args, network=origin_net)
# origin_net.load_state_dict(origin_net_data)


# 加载隐写模型
# qat_model_data = torch.load("jianzhi_stego_100bits.pth")
# qat_model_data = torch.load("jianzhi_stego_200bits.pth")
# qat_model_data = torch.load("jianzhi_stego_300bits.pth")
# qat_model_data = torch.load("jianzhi_stego_400bits.pth")
# qat_model_data = torch.load("jianzhi_stego_500bits.pth")
#
# saved_model_dir = 'trained_models/'
# float_model_file = 'check_point.pth'
# qat_model = load_model(saved_model_dir + float_model_file)
# qat_model = prune_network(args, network=qat_model)
# qat_model.load_state_dict(qat_model_data)
# qat_model = qat_model.to(torch.device('cpu'))


# kernel_size_height = qat_model.features[3].weight.shape[2]
# kernel_size_width = qat_model.features[3].weight.shape[3]
#
# feature = origin_net.features[3].weight
# P1 = torch.reshape(feature, (-1, kernel_size_height * kernel_size_width))
#
#
# feature = qat_model.features[3].weight
# Q1 = torch.reshape(feature, (-1, kernel_size_height * kernel_size_width))



print("stop")

# 累加器，self.data[0]存储判断正确的数量，self.data[1]存储总的数量
class Frequency:  #@save
    """在n个变量上累加"""
    def __init__(self, n = 100):  # [-1,1]范围内，n是取点数量，间隔为0.02
        """n=2时，self.data = [0.0, 0.0]"""
        self.n = n
        self.data = [0.0] * n
        self.center = round(n / 2)  # 代表0的点
        self.space = 2 / n
        self.x = []
        for i in range(n):
            self.x.append( (i-self.center) * self.space)


    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def frequency(self, tensor):
        shape = tensor.shape
        size = len(shape)

        if size == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    index = torch.round(tensor[i][j] / self.space) + self.center
                    if index > (self.n - 1):
                        index = torch.tensor(self.n - 1)
                    if index < 0:
                        index = torch.tensor(0)
                    self.data[int(index.item())] += 1

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# origin_frequency = Frequency()
# stego_frequency = Frequency()
#
#
# origin_frequency.frequency(P1)
# stego_frequency.frequency(Q1)
#
#
# p = torch.tensor(origin_frequency.data)
# q = torch.tensor(stego_frequency.data)
#
# for i in range(len(q)):
#     if p[i] != 0 and q[i] == 0:
#         q[i] = 0.0001
#
#
# p = p/p.sum()
# q = q/q.sum()
#
#
# print(p)
# print(q)
# x = origin_frequency.x
#
#
# # with open("jianzhi_x.txt","w") as f:
# #     for line in x:
# #         print(str(line))
# #         f.write(str(line) + '\n')
# #
# # with open("jianzhi_origin.txt","w") as f:
# #     for line in p:
# #         print(str(line.item()))
# #         f.write(str(line.item()) + '\n')
# #
# # with open("jianzhi_stego_100bit.txt","w") as f:
# #     for line in q:
# #         print(str(line.item()))
# #         f.write(str(line.item()) + '\n')
#
# # with open("jianzhi_stego_200bit.txt","w") as f:
# #     for line in q:
# #         print(str(line.item()))
# #         f.write(str(line.item()) + '\n')
#
# # with open("jianzhi_stego_300bit.txt","w") as f:
# #     for line in q:
# #         print(str(line.item()))
# #         f.write(str(line.item()) + '\n')
#
#
# divergence = F.kl_div(input=q.log(), target=p, reduction='sum')
# print("KL散度", divergence)
#
# print("stop")


"""
    定义解码网络
"""


class GcnNet(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GcnNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=self.use_bias)
        self.sig = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, conv1, i, is_quantitative_model = False):
        if is_quantitative_model:
            kernel_size_height = conv1.weight().shape[2]
            kernel_size_width = conv1.weight().shape[3]
            feature = conv1.weight().dequantize()
            feature = torch.reshape(feature, (-1, kernel_size_height * kernel_size_width))
            feature = torch.reshape(feature, (1, -1))

            """加入高斯分布噪声"""
            # gaosi = torch.normal(mean=0., std=i, size=feature.shape)
            # print(gaosi)
            # feature = feature + gaosi
            """加入均匀分布噪声"""
            # junyun= torch.rand(feature.shape) * 2 - 1
            # junyun = junyun * i
            # print(junyun.shape)
            # feature = feature + junyun

            logits = self.linear(feature)
            logits = self.sig(logits)
            logits = torch.reshape(logits, (-1, 1))

        else:
            kernel_size_height = conv1.weight.shape[2]
            kernel_size_width = conv1.weight.shape[3]
            feature = torch.reshape(conv1.weight, (-1, kernel_size_height * kernel_size_width))
            feature = torch.reshape(feature, (1, -1))

            """加入高斯分布噪声"""
            # gaosi = torch.normal(mean=0., std=i, size=feature.shape)
            # print(gaosi)
            # feature = feature + gaosi
            """加入均匀分布噪声"""
            # junyun= torch.rand(feature.shape) * 2 - 1
            # junyun = junyun * i
            # feature = feature + junyun

            logits = self.linear(feature)

            logits = self.sig(logits)
            logits = torch.reshape(logits, (-1, 1))

        return logits

gcn_model = torch.load("jianzhi_gcn_200bits.pth").to(torch.device("cpu"))

"""
    判决函数,网络输出结果进行判断分类
"""

def decision(secrets_hat):
    length0 = secrets_hat.shape[0]
    length1 = secrets_hat.shape[1]
    for i in range(length0):
        for j in range(length1):
            if secrets_hat[i][j] > 0.5:
                secrets_hat[i][j] = 1
            else:
                secrets_hat[i][j] = 0
    # 更新decision_hat
    decision_hat = secrets_hat
    return decision_hat

"""
    计算判决的准确率
"""

def gcn_accuracy(Y_decision, Y):
    print(Y_decision.shape)
    print(Y.shape)
    sum = Y_decision.numel()
    correct = 0
    error = 0
    length0 = Y_decision.shape[0]
    length1 = Y_decision.shape[1]
    for i in range(length0):
        for j in range(length1):
            if Y_decision[i][j] == Y[i][j]:
                correct += 1
            else:
                error += 1
    correct_rate = correct/sum
    error_rate = error/sum
    print("正确率：", correct_rate)
    print("错误率：", error_rate)
    return correct_rate, error_rate


# with open("jianzhi_gaosi_100bit.txt", "w") as f:
# with open("jianzhi_junyun_100bit.txt", "w") as f:
# with open("jianzhi_gaosi_200bit.txt", "w") as f:
# with open("jianzhi_junyun_200bit.txt", "w") as f:
# with open("jianzhi_gaosi_300bit.txt", "w") as f:
# with open("jianzhi_junyun_300bit.txt", "w") as f:
# with open("jianzhi_gaosi_400bit.txt", "w") as f:
# with open("jianzhi_junyun_400bit.txt", "w") as f:
# with open("jianzhi_gaosi_500bit.txt", "w") as f:
# with open("jianzhi_junyun_500bit.txt", "w") as f:
#     for i in range(1, 31):
#         i=i*0.01
#         print(i)
#         gcn_out = gcn_model(qat_model.features[3], i, is_quantitative_model=False)
#         # gcn_out = gcn_out[flag_point:, :]
#
#         Y_decision = decision(gcn_out)
#         correct_rate, error_rate = gcn_accuracy(Y_decision, secret_tensor)
#
#         f.write(str(error_rate) + '\n')



def ascii2str(result_binarys):
    all_binarys = []
    result_str = ""
    for binarys in result_binarys:
        all_binarys += binarys
    count = 0
    temp_str = ""
    for bit in all_binarys:
        temp_str += str(bit)
        count += 1
        if count % 8 == 0:
            # print(int(temp_str, 2), "对应字符：", chr(int(temp_str, 2)))
            result_str += chr(int(temp_str, 2))
            temp_str = ""
    return result_str


#for i in range(43):
for i in range(5):
    qat_model_data = torch.load("jianzhi_stego_200bits_jit"+str(i)+".pth")

    saved_model_dir = '../trained_models/'
    float_model_file = 'check_point.pth'
    qat_model = load_model(saved_model_dir + float_model_file)
    qat_model = prune_network(args, network=qat_model)
    qat_model.load_state_dict(qat_model_data.state_dict())
    qat_model = qat_model.to(torch.device("cpu"))
    gcn_out = gcn_model(qat_model.features[3], 0, is_quantitative_model=False)

    Y_decision = decision(gcn_out)
    Y_decision = Y_decision.reshape([1, -1])
    decoded = Y_decision.detach().numpy().tolist()[0]
    result_binarys = list(map(int, decoded))
    result_str = ascii2str([result_binarys])
    print(result_str, end="\n")
