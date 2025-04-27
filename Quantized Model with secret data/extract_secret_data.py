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
                nn.init.constant_(module.bias,0)


"""
------------------------------
    2、Helper functions
------------------------------
"""

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


# # 加载原始模型
#
# # origin_net_data = torch.load("lianghua_stego_0bits.pth")
# saved_model_dir = 'trained_models/'
# float_model_file = 'check_point.pth'
# origin_net = load_model(saved_model_dir + float_model_file)
# origin_net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# torch.quantization.prepare_qat(origin_net, inplace=True)
# origin_net = torch.quantization.convert(origin_net)
# # origin_net.load_state_dict(origin_net_data)
#
#
# # 加载隐写模型
# # qat_model_data = torch.load("lianghua_stego_100bits.pth")
# qat_model_data = torch.load("lianghua_stego_200bits_jit0.pth")
# # qat_model_data = torch.load("lianghua_stego_300bits.pth")
# # qat_model_data = torch.load("lianghua_stego_400bits.pth")
# # qat_model_data = torch.load("lianghua_stego_500bits.pth")
#
# saved_model_dir = 'trained_models/'
# float_model_file = 'check_point.pth'
# qat_model = load_model(saved_model_dir + float_model_file)
# qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# torch.quantization.prepare_qat(qat_model, inplace=True)
# qat_model = torch.quantization.convert(qat_model)
# qat_model.load_state_dict(qat_model_data)
#
#
#
#
#
# kernel_size_height = qat_model.features[3].weight().shape[2]
# kernel_size_width = qat_model.features[3].weight().shape[3]
#
# feature = origin_net.features[3].weight().dequantize()
# P1 = torch.reshape(feature, (-1, kernel_size_height * kernel_size_width))
#
#
# feature = qat_model.features[3].weight().dequantize()
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
                    # print(index)
                    if index > (self.n - 1):
                        # continue
                        index = torch.tensor(self.n - 1)
                    if index < 0:
                        # continue
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

# with open("lianghua_x.txt","w") as f:
#     for line in x:
#         print(str(line))
#         f.write(str(line) + '\n')
#
# with open("lianghua_origin.txt","w") as f:
#     for line in p:
#         print(str(line.item()))
#         f.write(str(line.item()) + '\n')
#
# with open("lianghua_stego_100bit.txt","w") as f:
#     for line in q:
#         print(str(line.item()))
#         f.write(str(line.item()) + '\n')
#
# with open("lianghua_stego_200bit.txt","w") as f:
#     for line in q:
#         print(str(line.item()))
#         f.write(str(line.item()) + '\n')

# with open("lianghua_stego_300bit.txt","w") as f:
#     for line in q:
#         print(str(line.item()))
#         f.write(str(line.item()) + '\n')
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
        # print(conv1.weight)
        if is_quantitative_model:
            kernel_size_height = conv1.weight().shape[2]
            kernel_size_width = conv1.weight().shape[3]
            feature = conv1.weight().dequantize()
            feature = torch.reshape(feature, (-1, kernel_size_height * kernel_size_width))
            feature = torch.reshape(feature, (1, -1))
            # print(feature.shape)

            """加入高斯分布噪声"""
            # gaosi = torch.normal(mean=0., std=i, size=feature.shape)
            # print(gaosi)
            # feature = feature + gaosi
            """加入均匀分布噪声"""
            # junyun= torch.rand(feature.shape) * 2 - 1
            # junyun = junyun * i
            # # print(junyun.shape)
            # feature = feature + junyun

            logits = self.linear(feature)

            logits = self.sig(logits)
            logits = torch.reshape(logits, (-1, 1))

        else:
            kernel_size_height = conv1.weight.shape[2]
            kernel_size_width = conv1.weight.shape[3]
            feature = torch.reshape(conv1.weight, (-1, kernel_size_height * kernel_size_width))
            feature = torch.reshape(feature, (1, -1))
            logits = self.linear(feature)
            logits = self.sig(logits)
            logits = torch.reshape(logits, (-1, 1))
        return logits


# gcn_model = torch.load("lianghua_gcn_100bits.pth")
gcn_model = torch.load("lianghua_gcn_200bits.pth")
# gcn_model = torch.load("lianghua_gcn_300bits.pth")
# gcn_model = torch.load("lianghua_gcn_400bits.pth")
# gcn_model = torch.load("lianghua_gcn_500bits.pth")


# secret_tensor = torch.load("lianghua_secret_100bits.pkl")
# secret_tensor = torch.load("lianghua_secret_200bits.pkl")
# secret_tensor = torch.load("lianghua_secret_300bits.pkl")
# secret_tensor = torch.load("lianghua_secret_400bits.pkl")
# secret_tensor = torch.load("lianghua_secret_500bits.pkl")




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
            # if i < 6:
            #     continue
            if Y_decision[i][j] == Y[i][j]:
                correct += 1
            else:
                error += 1
    correct_rate = correct/sum
    error_rate = error/sum
    print("正确率：", correct_rate)
    print("错误率：", error_rate)
    return correct_rate, error_rate


# with open("lianghua_gaosi_100bit.txt", "w") as f:
# with open("lianghua_junyun_100bit.txt", "w") as f:
# with open("lianghua_gaosi_200bit.txt", "w") as f:
# with open("lianghua_junyun_200bit.txt", "w") as f:
# with open("lianghua_gaosi_300bit.txt", "w") as f:
# with open("lianghua_junyun_300bit.txt", "w") as f:
# with open("lianghua_gaosi_400bit.txt", "w") as f:
# with open("lianghua_junyun_400bit.txt", "w") as f:
# with open("lianghua_gaosi_500bit.txt", "w") as f:
# with open("lianghua_junyun_500bit.txt", "w") as f:
#     for i in range(1, 31):
#         i=i*0.01
#         print(i)
#         gcn_out = gcn_model(qat_model.features[3], i, is_quantitative_model=True)
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
    qat_model_data = torch.load("lianghua_stego_200bits_jit"+str(i)+".pth")

    saved_model_dir = '../trained_models/'
    float_model_file = 'check_point.pth'
    qat_model = load_model(saved_model_dir + float_model_file)
    qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(qat_model, inplace=True)
    qat_model = torch.quantization.convert(qat_model)
    qat_model.load_state_dict(qat_model_data)

    gcn_out = gcn_model(qat_model.features[3], 0, is_quantitative_model=True)
    Y_decision = decision(gcn_out)
    Y_decision = Y_decision.reshape([1, -1])
    decoded = Y_decision.detach().numpy().tolist()[0]
    result_binarys = list(map(int, decoded))
    result_str = ascii2str([result_binarys])
    print(result_str,end="\n")





