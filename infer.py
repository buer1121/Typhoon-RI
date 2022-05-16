import torchvision.datasets
import numpy as np
from train import dct_by_block
from model_gmm import GMM
from tqdm import tqdm

# 导入测试数据
cifar10_test = torchvision.datasets.CIFAR10(
    root='datasets',
    train=False,
    download=False
)
test_data = [ ( np.array( item[0].convert("YCbCr") ), item[1] ) for item in cifar10_test ]

model_list = []
for index in range( 10 ):
    gm = GMM()
    gm.load( "model_ckpt/class-{}.gm".format( index ) )
    model_list.append( gm )

count, r_count, N = 0, 0, 4
for item in tqdm( test_data ):
    img_feat_list, label = dct_by_block( item[0] ), item[1]
    gm = GMM( 6 )
    gm.fit( img_feat_list )
    proba_list = [ sum( [ gm.predict_proba( img_feat ) * model_list[i].predict_proba( img_feat ) for img_feat in img_feat_list ] ) for i in range( 10 ) ]
    if label in np.argsort( proba_list )[-N:]:
        r_count += 1
    count += 1
print( "N={}, Acc={}".format( N, r_count / count ) )