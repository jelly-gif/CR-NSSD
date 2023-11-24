
import torch.nn.functional as F
from .Utils import *
from .tcn import *

import csv

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

class gatelayer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(gatelayer, self).__init__()

        self.L1 = nn.Sequential(
            nn.Linear(input_channel, output_channel),
        )

        self.L2 = nn.Sequential(
            nn.Linear(input_channel, output_channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        l1 = self.L1(x)
        l2 = self.L2(x)
        return torch.mul(l1, l2)

class IMILmask(nn.Module):
    def __init__(self):
        super(IMILmask, self).__init__()
        self.time_step = 12
        self.size = 256

        self.L1 = nn.Sequential(
            nn.Linear(self.size, self.size),
            nn.Tanh()
        )

        self.L2 = nn.Sequential(
            nn.Linear(self.size, self.size),
            nn.Sigmoid() #0~1

        )
        self.attention_weights = nn.Linear(self.size, self.size) #注意力权重
        return

    def forward(self, x):
        x = x.permute(0,2,1)
        l1 = self.L1(x)
        l2 = self.L2(x)
        A_imi = self.attention_weights(l2*l1) # element wise multiplication
        A_imi=torch.sigmoid(A_imi)
        x = torch.mul(x,A_imi)
        return x

class multiDomainSeqLayer(nn.Module):
    def __init__(self,unit, multi_layer, block_num=0):
        super(multiDomainSeqLayer, self).__init__()
        self.mask=IMILmask()
        self.unit = unit
        self.block_num = block_num
        self.multi = multi_layer
        self.scale_size=4
        self.processing1 = nn.Linear(self.multi, self.multi)
        self.processing2 = nn.Linear(self.multi, 1)

        self.relu = nn.ReLU()
        self.Gating = nn.ModuleList()
        self.output_channel = self.scale_size * self.multi #20
        self.Gating.append(gatelayer(self.scale_size, self.output_channel))
        self.Gating.append(gatelayer(self.output_channel, self.output_channel))
        self.Gating.append(gatelayer(self.output_channel, self.output_channel))

    def HighFrequencyRetentionBlock(self, input):

        batch_size, k, input_channel, block_num, time_step = input.size()
        input = input.view(batch_size, -1, block_num, time_step)
        frequencySeq = torch.rfft(input, 1, onesided=False)

        real = frequencySeq[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, block_num, -1) #32x140x(4x12)
        img = frequencySeq[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, block_num, -1)
        for i in range(3):
            real = self.Gating[i](real)
            img = self.Gating[i](img)

        real = real.reshape(batch_size, block_num, self.scale_size, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, block_num, self.scale_size, -1).permute(0, 2, 1, 3).contiguous()

        HighFrequencySeq = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)

        HighFrequencySeq=torch.sum(HighFrequencySeq,-1)

        return HighFrequencySeq

    def forward(self, x, mul_L):

        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        spectralSeq = torch.matmul(mul_L, x)
        HightSpectralSeq = self.HighFrequencyRetentionBlock(spectralSeq).unsqueeze(2)
        HightSpectralSeq = torch.sum(HightSpectralSeq, dim=1)
        BindingSiteSeq0 = self.processing1(HightSpectralSeq).squeeze(1)
        BindingSiteSeq = self.processing2(BindingSiteSeq0) #32x100x12 在此处增加一个mask层
        BindingSiteSeq=self.mask(BindingSiteSeq)
        BindingSiteSeq=BindingSiteSeq.permute(0,2,1)

        return BindingSiteSeq


class Model(nn.Module):
    def __init__(self, units, block_num, multi_layer,dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        self.unit = units #特征维度 100
        self.block_num = block_num
        self.unit = units
        self.alpha = leaky_rate
        self.weight_graph = nn.Parameter(torch.zeros(size=(1,1,1, 1)))
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.ml = nn.Linear(1, self.unit)
        self.tcn =TemporalConvNet(self.unit,self.unit)
        self.multi_layer = multi_layer

        self.bn1 = nn.BatchNorm1d(256)

        self.seqGraphBlock = nn.ModuleList()
        self.seqGraphBlock.extend(
            [multiDomainSeqLayer(self.unit, self.multi_layer, block_num=i) for i in range(self.block_num)])

        self.fc_shape = nn.Sequential(
            nn.Linear(41, int(self.unit)),
            nn.LeakyReLU(),
            nn.Linear(int(self.unit), int(self.unit)),
        )
        self.Ifc_shape = nn.Sequential(
            nn.Linear(int(self.unit), int(self.unit)),
            nn.LeakyReLU(),
            nn.Linear(int(self.unit),41),
        )
        self.fc_prob = nn.Sequential( #亲和力约束12->12,100->1 32x12x100
            nn.Linear(int(self.unit), self.unit),
            nn.LeakyReLU(),
            nn.Linear(int(self.unit), 1),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def seq_graph_ing(self, x,input_prob):
        input=x.repeat(1,256,1)
        attention = self.district_graph_attention(input,input_prob)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat)) #得到拉普拉斯矩阵，类似GCN
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def district_graph_attention(self, input,input_prob):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        input_prob=normalized_input(input_prob)
        L=torch.mean(input_prob,1)
        L=L.unsqueeze(1)
        attention=attention*(L)
        attention = self.dropout(attention)

        return attention

    def forward(self, x,input_prob):
        x=self.fc_shape(x)
        input_prob=self.fc_shape(input_prob)
        mul_L, attention = self.seq_graph_ing(x,input_prob)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for stack_i in range(self.block_num):
            forecast = self.seqGraphBlock[stack_i](X, mul_L)
            result.append(forecast)

        forecast = result[0]+result[1]
        forecast_site_prob=forecast
        forecast_feature=forecast_site_prob.permute(0, 2, 1).contiguous().view(-1,self.unit)
        forecast_site_prob=self.bn1(forecast_site_prob)
        forecast_site_prob=torch.sigmoid(self.fc_prob(forecast_site_prob.permute(0, 2, 1))).contiguous()
        forecast_site_prob=torch.squeeze(forecast_site_prob,2)
        forecast=forecast.permute(0, 2, 1).contiguous()

        forecast=self.Ifc_shape(forecast)

        return forecast_site_prob,forecast_feature,forecast