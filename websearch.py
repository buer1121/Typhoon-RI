import cv2
import numpy as np
import torchvision.datasets
from GMM import GMM
from tqdm import tqdm
from multiprocessing import Process

# 基于 DCT 的特征提取
def dct_by_block( img, block_size=8, stride=2, compress_size=4 ):
    img = np.float32( img )
    height, width, channel = img.shape
    img_feat_list = []
    for h_end in range( block_size, height+2, stride ):                                   #8->32,++2
        for w_end in range( block_size, width+2, stride ):                                #8->32,++2
            img_feat = np.zeros( ( compress_size * compress_size * channel ) )            #4*4*3*[0]
            # print("img_feat",img_feat)
            for c in range( channel ):                                                  #每个通道4*4作DCT后展平
                img_block = img[ h_end-block_size:h_end, w_end-block_size:w_end, c ]
                # print(cv2.dct( img_block ),cv2.dct( img_block ).shape)
                # print(cv2.dct( img_block )[:compess_size,:compess_size,],cv2.dct( img_block )[:compess_size,:compess_size,].shape)
                # print(cv2.dct( img_block )[:compess_size,:compess_size,].flatten(),cv2.dct( img_block )[:compess_size,:compess_size,].flatten().shape)
                img_feat[ compress_size * compress_size * c:compress_size * compress_size * (c+1) ] = cv2.dct( img_block )[:compress_size,:compress_size,].flatten()
                # print(img_feat)
            img_feat_list.append( img_feat )
    return img_feat_list

def train_task( dataset_by_class , class_index ):
    model = GMM( 10 )
    model.fit( dataset_by_class[class_index] )
    model.save( "model_save/class-{}.gm".format( class_index ) )


def train():
    # 导入数据
    cifar10 = torchvision.datasets.CIFAR10(
                                        root='dataset',
                                        train=True,
                                        download=True)

    # print(cifar10)
    train_data = [ ( np.array( item[0].convert("YCbCr") ), item[1] ) for item in cifar10 ]

    # 开始训练
    class_sample_size = [5000] * 10
    count_list = [0] * 10
    dataset_by_class = [ [] for i in range(10) ]
    print("class_sample_size",class_sample_size)
    print("count_list",count_list)
    print("dataset_by_class",dataset_by_class)
    for item in tqdm( train_data ):
        class_index = item[1]
        if count_list[class_index] < class_sample_size[class_index]:
            dataset_by_class[class_index].extend( dct_by_block( item[0] ) )
            count_list[class_index] += 1
    print("dataset_by_class",dataset_by_class[0][0].shape,len(dataset_by_class[0]),len(dataset_by_class))

    model_list = []
    for class_index, _ in enumerate( dataset_by_class ):
        p = Process( target=train_task, args=(dataset_by_class, class_index, ) )
        p.start()
        p.join()


def test():
    # 导入测试数据
    cifar10_test = torchvision.datasets.CIFAR10(
        root='dataset',
        train=False,
        download=True
    )
    test_data = [ ( np.array( item[0].convert("YCbCr") ), item[1] ) for item in cifar10_test ]

    model_list = []
    for index in range( 10 ):
        gm = GMM()
        gm.load( "model_save/class-{}.gm".format( index ) )
        model_list.append( gm )

    count, r_count, N = 0, 0, 5
    for item in tqdm( test_data ):
        img_feat_list, label = dct_by_block( item[0] ), item[1]
        gm = GMM( 6 )
        gm.fit( img_feat_list )
        proba_list = [ sum( [ gm.predict_proba( img_feat ) * model_list[i].predict_proba( img_feat ) for img_feat in img_feat_list ] ) for i in range( 10 ) ]
        if label in np.argsort( proba_list )[-N:]:
            r_count += 1
        count += 1
    print( "N={}, Acc={}".format( N, r_count / count ) )

    return N


if __name__ == '__main__':
    #开始训练
    train()
    #开始测试
    N=test()
