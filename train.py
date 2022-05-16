import cv2
import numpy as np
import torchvision.datasets
from model_gmm import GMM
from tqdm import tqdm
from multiprocessing import Process
    
# 导入数据shi   d
cifar10 = torchvision.datasets.CIFAR10(
    root='datasets',
    train=True,
    download=True
)
train_data = [ ( np.array( item[0].convert("YCbCr") ), item[1] ) for item in cifar10 ]

# 基于 DCT 的特征提取
def dct_by_block( img, block_size=8, stride=2, compess_size=4 ):
    img = np.float32( img )
    height, width, channel = img.shape
    img_feat_list = []
    for h_end in range( block_size, height, stride ):
        for w_end in range( block_size, width, stride ):
            img_feat = np.zeros( ( compess_size * compess_size * channel ) )
            for c in range( channel ):
                img_block = img[ h_end-block_size:h_end, w_end-block_size:w_end, c ]
                img_feat[ compess_size * compess_size * c:compess_size * compess_size * (c+1) ] = cv2.dct( img_block )[:compess_size,:compess_size,].flatten()
            img_feat_list.append( img_feat )
    return img_feat_list

# 开始训练
class_sample_size = [5000] * 10
count_list = [0] * 10
dataset_by_class = [ [] for i in range(10) ]

for item in tqdm( train_data ):
    class_index = item[1]
    if count_list[class_index] < class_sample_size[class_index]:
        dataset_by_class[class_index].extend( dct_by_block( item[0] ) )
        count_list[class_index] += 1

def train_task( dataset_by_class , class_index ):
    model = GMM( 10 )
    model.fit( dataset_by_class[class_index] )
    model.save( "model_ckpt/class-{}.gm".format( class_index ) )

model_list = []
for class_index, _ in enumerate( dataset_by_class ):
    p = Process( target=train_task, args=(dataset_by_class, class_index, ) )
    p.start()
    p.join()