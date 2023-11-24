import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd
import csv
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:          # 注意表头
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
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

def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            # print(counter)
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df,seq_size=41,normalize_method=None, norm_statistic=None, interval=1):
        self.interval = interval
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        self.seq_size=seq_size
        # if normalize_method:
        #     self.data, _ = normalized(self.data, normalize_method, norm_statistic)
    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - 1
        train_data = self.data[lo: hi]
        # print('df.shape '+str(self.data.shape))

        #结合位点、时序训练数据
        train_seq,train_label=self.get_seq_label(train_data)
        # print(type(train_seq[0]))
        # print(type(train_label[0]))
        train_seq = torch.from_numpy(train_seq).type(torch.float)
        train_label = torch.from_numpy(train_label).type(torch.float)


        return train_seq,train_label  #train+target

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(1, self.df_length + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

    def get_seq_label(self,seq):
        seqs=[]
        labels=[]
        for ele in seq:
            i=0
            while i<len(ele):
                ele[i]=float(ele[i])
                i+=1
            temp_seq=ele[:self.seq_size]
            temp_label=ele[self.seq_size]
            seqs.append(temp_seq)
            labels.append(temp_label)
        return np.array(seqs,dtype='float64'),np.array(labels)

#数据读取测试
if __name__ == '__main__':
   print("done!")
