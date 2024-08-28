import torch
import numpy as np


class MyDataset():
    def __init__(self,data_path,window_size=6) -> None:
        self.window_size = window_size
        self.data = torch.from_numpy(self.loading(data_path)).float()

    def loading(self,data_path):
        '''
        预处理
        '''
        data = np.load(data_path)['volume']
        # print(data.shape)
        # 获取最大最小值
        self.max_val, self.min_val = np.max(data), np.min(data)
        # 划分窗口
        dataset = slidingWindow(data,self.window_size)
        # 改变数据维度
        dataset = np.array(dataset).transpose(0,1,4,2,3) # 本来是(1914, 6, 10, 20, 2),经过此处变为(1914, 6, 2, 10, 20)
        dataset = dataset.reshape(dataset.shape[0],dataset.shape[1],-1) # (1914, 6, 400)
        # 归一化
        dataset = (dataset - self.min_val) / (self.max_val - self.min_val)
        return dataset
     
    def denormalize(self,x):    
        '''
        反归一化
        '''    
        return x * (self.max_val - self.min_val) + self.min_val

def slidingWindow(seqs,size):
    """
    seqs: ndarray sequence, shape=(序列长度,地区数量,2) 但实际上是4维
    size: sliding window size
    """
    result = []
    for i in range(seqs.shape[0] - size + 1):
        result.append(seqs[i:i + size,:,:,:]) #(6, 10, 20, 2) 
    return result


if __name__ == "__main__":
    train_path = 'dataset/volume_train.npz'
    dataset = MyDataset(train_path)
    print(dataset.data.shape)
    test_path = 'dataset/volume_test.npz'
    dataset = MyDataset(test_path)
    print(dataset.data.shape)