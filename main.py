import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from datetime import datetime
from models.handler import train
import argparse
from data_loader.SiteBinding_dataloader import *

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='m6A_data')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--batch_size', type=int, default=145)
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--decay_rate', type=float, default=0.1) #0.5
parser.add_argument('--dropout_rate', type=float, default=0.5) #0.5

args = parser.parse_args()
print(f'Training configs: {args}')

result_train_file = os.path.join('output', args.dataset, 'M41')
result_test_file = os.path.join('output', args.dataset, 'test')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)

if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            i=0
            all_result=[]
            while i<10:
                print('fold '+str(i)+' ')
                print('-'*99)
                train_data = []
                valid_data = []
                ReadMyCsv(train_data,'./utils/0.RNA_m_process/m6A/example/Train'+str(i)+'.csv')
                ReadMyCsv(valid_data,'./utils/0.RNA_m_process/m6A/example/Test'+str(i)+'.csv')
                print('Train begining!')
                forecast_feature,result=train(train_data, valid_data, args, result_train_file,i)
                all_result.append(result)
                i+=1
            after_train = datetime.now().timestamp()
            print(all_result)
            print(f'Training took {(after_train - before_train) / 60} minutes')

        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')




