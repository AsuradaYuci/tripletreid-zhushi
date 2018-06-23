#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import cv2
import pickle
import sys

#sys.argv将程序本身和给程序参数返回一个list,这个list中的索引为0即sys.argv[0]就是程序本身
#sys.argv就是一个从程序外部获取参数的桥梁
if len(sys.argv) != 3:
    print("Usage: {} IMAGE_LIST_FILE MODEL_WEIGHT_FILE".format(sys.argv[0]))
    sys.exit(1)  #exit(1)：有错误退出，告诉解释器

# Specify the path to a Market-1501 image that should be embedded and the location of the weights we provided.
#指定应该嵌入的Market-1501图片的路径以及我们提供的权重的位置。
image_list = list(map(str.strip, open(sys.argv[1]).readlines()))  #str.strip()就是把这个字符串头和尾的空格，以及位于头尾的\n \t之类给删掉
#open(sys.argv[1])从命令行读取文件名,.readlines()读出所有行到一个列表.map 将list的内容映射为字符串str
weight_fname = sys.argv[2]
#命令行的第三个内容


# Setup the pretrained ResNet

#This is based on the Lasagne ResNet-50 example with slight modifications to allow for different input sizes.
#The original can be found at: https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/ImageNet%20Pretrained%20Network%20(ResNet-50).ipynb
#Lasagne是在Theano的基础上封装的架构，Theano实现的是关于矩阵的运算，而Lasagne在Theano的基础上定义了layer的概念
import theano
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax


def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
       创建堆叠的Lasagne layers  ,ConvLayer - > BN - >（ReLu）
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer    传入层:Lasagne layer 的实例   Parent layer 父层
        Parent layer

    names : list of string              字符串列表
        Names of the layers in block   块中的层名称

    num_filters : int                           卷积层中的滤波器数量
        Number of filters in convolution layer

    filter_size : int
        Size of filters in convolution layer    卷积层中滤波器的大小

    stride : int
        Stride of convolution layer             卷积层的步幅

    pad : int
        Padding of convolution layer             卷积层的填充

    use_bias : bool
        Whether to use bias in conlovution layer  是否在卷积层中使用偏差bias

    nonlin : function
        Nonlinearity type of Nonlinearity layer  非线性层的非线性类型

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers   堆积层的字典
        last_layer_name : string
            Last layer name            最后一层名称
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, stride, pad,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))
    #net字典{第一个卷积层的名称，卷积层的参数}

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    #{第二个批量归一化层的名称，net[-1][1]是上一个层的输出，ConvLayer（）
    if nonlin is not None:
        #存在非线性
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))
        #{非线性层的名称，net[-1][1]是上一个层的输出BatchNormLayer（）

    return dict(net), net[-1][0]
    #返回，net字典，最后一层的名称

def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block  创建双分支的残差块

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer   传入层:Lasagne layer 的实例   Parent layer 父层
        Parent layer

    ratio_n_filter : float                                          卷积核组在输入残差块时的比例因子
        Scale factor of filter bank at the input of residual block

    ratio_size : float                                     卷积核尺寸的比例因子
        Scale factor of filter size

    has_left_branch : bool                                  如果为True，则左分支包含简单块
        if True, then left branch contains simple block

    upscale_factor : float                                         卷积核组在残差块输出处的比例因子
        Scale factor of filter bank at the output of residual block

    ix : int                              残差块的ID
        Id of residual block

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch 右分支
    #def build_simple_block(incoming_layer, names, num_filters, filter_size, stride, pad, use_bias=False, nonlin=rectify)
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, list(map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern)),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    #names=list(map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern))
    #num_filters=int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter),ratio_n_filter=1
    #filter_size = 1,stride = 1, pad = 0
    net.update(net_tmp)
    # net_tmp添加到指定字典dict里的字典

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    #这一层的输入层 = 上一层的输出 net[last_layer_name]
    # names = list(map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern))
    # num_filters = lasagne.layers.get_output_shape(net[last_layer_name])[1]
    # filter_size = 3,stride = 1, pad = 1
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    #这一层的输入层 = 上一层的输出 net[last_layer_name]
    # names = list(map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern))
    #num_filters =  lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor=4
    #filter_size = 1,stride = 1, pad = 0

    net.update(net_tmp)

    right_tail = net[last_layer_name]
    #右分支的尾部 = 上一层的输出 net[last_layer_name]
    left_tail = incoming_layer
    #左分支的尾部 = 输入层

    # left branch  左分支  默认为has_left_branch=False
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, list(map(lambda s: s % (ix, 1, ''), simple_block_name_pattern)),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        #names = list(map(lambda s: s % (ix, 1, ''), simple_block_name_pattern))
        #num_filters =  int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter) ratio_n_filter=1
        #filter_size = 1,stride =int(1.0/ratio_size), pad = 0
        net.update(net_tmp)
        left_tail = net[last_layer_name]
        ##左分支的尾部 = 上一层的输出 net[last_layer_name]
    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    #ElemwiseSumLayer该图层执行其输入图层的元素总和。它要求所有输入图层具有相同的输出形状。
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify, name = 'res%s_relu' % ix)

    return net, 'res%s_relu' % ix
    #'res%s_relu' % ix = Last layer name

def build_model(input_size):
    net = {}
    net['input'] = InputLayer(input_size)
    #输入层，例如InputLayer((100, 20))

    # def build_simple_block(incoming_layer, names, num_filters, filter_size, stride, pad, use_bias=False, nonlin=rectify)
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 2, 3, use_bias=True)
    #输入层：net['input']，
    #names = ['conv1', 'bn_conv1', 'conv1_relu']
    #num_filters = 64，filter_size = 7, stride = 2, pad = 3,
    net.update(sub_net)

    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    #池化层，

    #def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,upscale_factor=4, ix=''):
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)

    return net


#Setup the original network                       设置原始网络
resnet = build_model(input_size=(None, 3, 256,128))

#Now we modify the network's final pooling layer and add 2 new layers at the end to predict the 128-dimensional embedding.
#现在我们修改网络的最终池化层，并在最后添加2个新层来预测128维嵌入。
#Different input size.不同的输入大小。
inp = resnet['input']

network_features = resnet['pool5']

network_features.pool_size=(8,4)

#New additional final layer    新增加最后一层
network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
        network_features,
        num_units=1024,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform('relu'),
        b=None))
#lasagne.layers.batch_norm将批量归一化应用于现有图层。
#lasagne.layers.DenseLayer(incoming, num_units,W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
#   nonlinearity=lasagne.nonlinearities.rectify, num_leading_axes=1, **kwargs)

network_out = lasagne.layers.DenseLayer(
        network,
        num_units=128,
        nonlinearity=None,
        W=lasagne.init.Orthogonal())



#Setup the function to predict the embeddings.     设置预测嵌入的函数。
predict_features = theano.function(
            inputs=[inp.input_var],
            outputs=lasagne.layers.get_output(network_out, deterministic=True))
#function是一个由inputs计算outputs的对象，它关于怎么计算的定义一般在outputs里面，这里outputs一般会是一个符号表达式。
#inputs:输入是一个python的列表list，里面存放的是将要传递给outputs的参数，这里inputs不是共享变量shared variables.
#outputs: 输出是一个存放变量的列表list或者字典dict
#lasagne.layers.get_output计算一个或多个给定层的网络输出。

#Set the parameters             设置参数
with np.load(weight_fname) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network_out, param_values)



#We subtract the per-channel mean of the "mean image" as loaded from the original ResNet-50 weight dump.
#我们减去从原始ResNet-50权重转储装载的“平均图像”的每通道平均值。
#For simplcity, we just hardcode it here.为了简化，我们只是在这里硬编码。
im_mean = np.asarray([103.0626238, 115.90288257, 123.15163084], dtype=np.float32)



# a little helper function to create a test-time augmentation batch.一个辅助函数来创建一个测试时间增强批次。
def get_augmentation_batch(image, im_mean):
    #Resize it correctly, as needed by the test time augmentation.
    #根据测试时间增加的需要正确调整它的大小。
    image = cv2.resize(image, (128+16, 256+32))

    #Change into CHW format
    image = np.rollaxis(image,2)
#调用 np.rollaxis(a,2)函数意思就是将2轴旋转至轴0的前面，轴序0，1，2变成1,2,0

    #Setup storage for the batch              批次的设置存储
    batch = np.zeros((10,3,256,128), dtype=np.float32)

    #Four corner crops and the center crop        四角裁剪和中心裁剪
    batch[0] = image[:,16:-16, 8:-8]    #Center crop
    batch[1] = image[:,   :-32,   :-16] #Top left
    batch[2] = image[:,   :-32, 16:]    #Top right
    batch[3] = image[:, 32:,      :-16] #Bottom left
    batch[4] = image[:, 32:,    16:]    #Bottom right

    #Flipping  翻转
    batch[5:] = batch[:5,:,:,::-1]

    #Subtract the mean 减去平均值
    batch = batch-im_mean[None,:,None,None]

    return batch



for image_filename in image_list:
    print(image_filename, end=",")
    sys.stdout.flush()#加入sys.stdout.flush()才能一秒输一个数字，刷新输出

    image = cv2.imread(image_filename)#读图
    if image is None:
        raise ValueError("Couldn't load image {}".format(image_filename))

    #Setup a batch of images and use the function to predict the embedding.设置一批图像并使用该函数预测嵌入。
    batch = get_augmentation_batch(image, im_mean)
    embedding = np.mean(predict_features(batch), axis=0)
    #axis=0,每一列的平均值
    print(','.join(map(str, embedding)))
    #','.join 用于将序列中的元素以指定的字符(这里是‘，’）连接生成一个新的字符串
    #map(str, embedding)，将embedding映射为字符串格式
