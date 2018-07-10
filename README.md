# Triplet-based Person Re-Identification
#只是进行了注释，删去了原始代码版本中的一些文件，若要实现整个程序，请参考原始代码版本
Code for reproducing the results of our [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737) paper.

We provide the following things:我们提供以下内容：
- The exact pre-trained weights for the TriNet model as used in the paper, including some rudimentary example code for using it to compute embeddings.
#本文中使用的TriNet模型的精确预训练权重，包括使用它计算嵌入的一些基本示例代码。
  See section [Pretrained models](#pretrained-models).
  #请参见[预训练模型]一节（＃pretrained-models）。
  
- A clean re-implementation of the training code that can be used for training your own models/data.
#干净的重新实施训练的代码，可用于训练自己的模型/数据。
  See section [Training your own models](#training-your-own-models).
  #请参见[Training your own models]（＃training-your-own-models）。
  
- A script for evaluation which computes the CMC and mAP of embeddings in an HDF5 ("new .mat") file.
#用于评估的脚本，在嵌入的HDF5（“new .mat”）文件中计算CMC和mAP。

  See section [Evaluating embeddings](#evaluating-embeddings).
  #请参见[Evaluating embeddings]（＃evaluation-embeddings）一节。
  
- A list of [independent re-implementations](#independent-re-implementations).
#[independent re-implementations]的列表（独立重新实现）。

If you use any of the provided code, please cite:
```
@article{HermansBeyer2017Arxiv,
  title       = {{In Defense of the Triplet Loss for Person Re-Identification}},
  author      = {Hermans*, Alexander and Beyer*, Lucas and Leibe, Bastian},
  journal     = {arXiv preprint arXiv:1703.07737},
  year        = {2017}
}
```


# Pretrained TensorFlow models

For convenience, we provide the pretrained weights for our TriNet TensorFlow model, 
#为了方便起见，我们为TriNet TensorFlow模型提供了预训练的权重，
trained on Market-1501 using the code from this repository and the settings form our paper. 
#在Market-1501上使用此存储库中的代码和本文的设置进行了培训。
The TensorFlow checkpoint can be downloaded in the [release section](https://github.com/VisualComputingInstitute/triplet-reid/releases/tag/250eb1).


# Pretrained Theano models

We provide the exact TriNet model used in the paper, which was implemented in
[Theano](http://deeplearning.net/software/theano/install.html)
and
[Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html).

As a first step, download either of these pre-trained models:
#作为第一步，请下载这些预先训练的模型中的任何一个：
- [TriNet trained on MARS](https://omnomnom.vision.rwth-aachen.de/data/trinet-mars.npz) (md5sum: `72fafa2ee9aa3765f038d06e8dd8ef4b`)
- [TriNet trained on Market1501](https://omnomnom.vision.rwth-aachen.de/data/trinet-market1501.npz) (md5sum: `5353f95d1489536129ec14638aded3c7`)

Next, create a file (`files.txt`) which contains the full path to the image files you want to embed, one filename per line, like so:
#接下来，创建一个文件（`files.txt`），其中包含要嵌入的图像文件的完整路径，每行一个文件名，如下所示：
```
/path/to/file1.png
/path/to/file2.jpg
```

Finally, run the `trinet_embed.py` script, passing both the above file and the pretrained model file you want to use, like so:
#最后，运行`trinet_embed.py`脚本，传递上面的文件和想要使用的预训练模型文件，如下所示：
```
python trinet_embed.py files.txt /path/to/trinet-mars.npz
```

And it will output one comma-separated line for each file, containing the filename followed by the embedding, like so:
#它会为每个文件输出一个以逗号分隔的行，包含文件名，然后是嵌入，如下所示：
```
/path/to/file1.png,-1.234,5.678,...
/path/to/file2.jpg,9.876,-1.234,...
```

You could for example redirect it to a file for further processing:
#例如，你可以将它重定向到一个文件进行进一步处理：
```
python trinet_embed.py files.txt /path/to/trinet-market1501.npz >embeddings.csv
```

You can now do meaningful work by comparing these embeddings using the Euclidean distance, for example, try some K-means clustering!
#您现在可以通过使用欧几里得距离比较这些嵌入来做有意义的工作，例如，尝试一些K均值聚类！
A couple notes:
- The script depends on [Theano](http://deeplearning.net/software/theano/install.html), 
[Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html) and 
[OpenCV Python](http://opencv.org/) (`pip install opencv-python`) being correctly installed.
- The input files should be crops of a full person standing upright, and they will be resized to `288x144` before being passed to the network.
#输入文件应该修剪为一个完整的直立人，在传递到网络之前它们将被调整为“288x144”。

# Training your own models

If you want more flexibility, we now provide code for training your own models.
#如果您想要更多的灵活性，我们现在提供用于培训您自己的模型的代码。
This is not the code that was used in the paper (which became a unusable mess),
#这不是在论文中使用的代码（它变得不可用），
but rather a clean re-implementation of it in [TensorFlow](https://www.tensorflow.org/),
#而是在[TensorFlow]（https://www.tensorflow.org/）中重新实现它，
achieving about the same performance.
#达到大致相同的性能。

- **This repository requires at least version 1.4 of TensorFlow.**
#**此版本库至少需要TensorFlow的1.4版本。**
- **The TensorFlow code is Python 3 only and won't work in Python 2!**

:boom: :fire: :exclamation: **If you train on a very different dataset, don't forget to tune the learning-rate** :exclamation: :fire: :boom:
#**如果您训练的数据集非常不同，请不要忘记调整学习速率**

## Defining a dataset 定义数据集

A dataset consists of two things: 
#数据集包含两件事情：

1. An `image_root` folder which contains all images, possibly in sub-folders.
#包含所有图像的`image_root`文件夹，可能位于子文件夹中。

2. A dataset `.csv` file describing the dataset.
#描述数据集的数据集`.csv`文件。

To create a dataset, you simply create a new `.csv` file for it of the following form:
#要创建数据集，只需为其创建一个新的`.csv`文件，格式如下：

```
identity,relative_path/to/image.jpg
```

Where the `identity` is also often called `PID` (`P`erson `ID`entity) and corresponds to the "class name",
#在“identity”通常也被称为“PID”（'P`erson` ID`entity）并且对应于“class 名称”的地方，
it can be any arbitrary string, but should be the same for images belonging to the same identity.
#它可以是任意的字符串，但对于属于相同标识的图像应该是相同的。

The `relative_path/to/image.jpg` is relative to aforementioned `image_root`.
#`relative_path / to / image.jpg`与前面提到的`image_root`相关。

## Training

Given the dataset file, and the `image_root`, you can already train a model.
#给定数据集文件和`image_root`，你可以训练一个模型。
The minimal way of training a model is to just call `train.py` in the following way:
#训练模型的最简单方法是用下面的方式调用`train.py`：


```
python train.py \
    --train_set data/market1501_train.csv \
    --image_root /absolute/image/root \
    --experiment_root ~/experiments/my_experiment
```

This will start training with all default parameters.
#这将开始使用所有默认参数进行训练。

We recommend writing a script file similar to `market1501_train.sh` where you define all kinds of parameters,
#我们建议编写一个类似于`market1501_train.sh`的脚本文件，在其中定义各种参数，

it is **highly recommended** you tune hyperparameters such as `net_input_{height,width}`, `learning_rate`,
`decay_start_iteration`, and many more.
#它是强烈推荐的**你可以调整超参数，比如`net_input_ {height，width}`，`learning_rate`，`decay_start_iteration`等等。

See the top of `train.py` for a list of all parameters.
#查看`train.py`的顶部列出所有参数。

As a convenience, we store all the parameters that were used for a run in `experiment_root/args.json`.
#为了方便起见，我们将所有用于运行的参数存储在`experiment_root / args.json`中。

### Pre-trained initialization  预先训练的初始化

If you want to initialize the model using pre-trained weights, such as done for TriNet,
you need to specify the location of the checkpoint file through `--initial_checkpoint`.
#如果您想使用预先训练的权重初始化模型，例如TriNet，您需要通过`--initial_checkpoint`指定checkpoint文件的位置。

For most common models, you can download the [checkpoints provided by Google here]#对于大多数常见的模型，您可以下载[这里由Google提供的检查点]
(https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

For example, that's where we get our ResNet50 pre-trained weights from,
#例如，这就是我们获得ResNet50预先训练的权重的地方，
and what you should pass as second parameter to `market1501_train.sh`.
#以及你应该传递给`market1501_train.sh`的第二个参数。

### Example training log  示例训练日志

This is what a healthy training on Market1501 looks like, using the provided script:
#这是Market1501上的一项健康训练，使用提供的脚本：

![Screenshot of tensorboard of a healthy Market1501 run](healthy-market-run.png)

The `Histograms` tab in tensorboard also shows some interesting logs.
#张量板中的“Histograms”选项卡也显示一些有趣的日志。

## Interrupting and resuming training  中断和恢复训练

Since training can take quite a while, interrupting and resuming training is important.
#由于训练需要相当长的一段时间，所以中断和恢复训练非常重要。

You can interrupt training at any time by hitting `Ctrl+C` or sending `SIGINT (2)` or `SIGTERM (15)`
to the training process;
#您可以随时通过按Ctrl + C或发送SIGINT（2）或SIGTERM（15）到训练过程来中断训练;

it will finish the current batch, store the model and optimizer state,
and then terminate cleanly.
#它会完成当前批处理，存储模型和优化器状态，然后干净地终止。

Because of the `args.json` file, you can later resume that run simply by running:
#由于`args.json`文件，您可以稍后恢复运行，只需运行：
```
python train.py --experiment_root ~/experiments/my_experiment --resume
```

The last checkpoint is determined automatically by TensorFlow using the contents of the `checkpoint` file.
#最后的checkpoint由TensorFlow使用`checkpoint`文件的内容自动确定。

## Performance issues  性能问题

For some reason, current TensorFlow is known to have inconsistent performance and can sometimes become very slow.
#出于某种原因，目前的TensorFlow已知具有不一致的性能，并且有时可能变得非常慢。
The current only known workaround is to install google's performance-tools and preload tcmalloc:
#目前唯一已知的解决方法是安装谷歌的性能工具并预加载tcmalloc：

```
env LD_PRELOAD=/usr/lib/libtcmalloc_minimal.so.4 python train.py ...
```
This fixes the issues for us most of the time, but not always.
#这大多数时间为我们解决了这些问题，但并非总是如此。
If you know more, please open an issue and let us know!

## Out of memory  内存不足

The setup as described in the paper requires a high-end GPU with a lot of memory.
#本文所述的设置需要高端GPU具有大量内存
If you don't have that, you can still train a model, but you should either use a smaller network,
#如果你没有这个，你仍然可以训练一个模型，但是你应该使用一个更小的网络，
or adjust the batch-size, which itself also adjusts learning difficulty, which might change results.
#或者调整批量大小，这本身也会调整学习难度，这可能会改变结果。

The two arguments for playing with the batch-size are `--batch_p` which controls the number of distinct
persons in a batch, and `--batch_k` which controls the number of pictures per person.
#使用批处理大小设置的两个参数是“--batch_p”，它控制批处理中不同人物的数量，“--batch_k”控制每个人的照片数量。
We usually lower `batch_p` first. 
 #我们通常首先降低`batch_p`。

## Custom network architecture  自定义网络架构

TODO: Documentation. It's also pretty straightforward.
#TODO：文档。 这也很简单。
### The core network  核心网络

### The network head

## Computing embeddings  计算嵌入

Given a trained net, one often wants to compute the embeddings of a set of pictures for further processing.
#给定一个训练好的网络，人们通常想要计算一组图片的嵌入以便进一步处理。
This can be done with the `embed.py` script, which can also serve as inspiration for using a trained model in a larger program.
#这可以通过`embed.py`脚本来完成，该脚本也可以作为在较大程序中使用训练模型的灵感。

The following invocation computes the embeddings of the Market1501 query set using some network:
#以下调用使用某个网络计算Market1501查询集的嵌入：
```
python embed.py \
    --experiment_root ~/experiments/my_experiment \
    --dataset data/market1501_query.csv \
    --filename test_embeddings.h5
```
 python embed.py 
--experiment_root checkpoint
--dataset data/market1501_test.csv 
--filename market1501_test_embeddings.h5

python embed.py 
--experiment_root checkpoint 
--dataset data/market1501_query.csv 
--filename market1501_query_embeddings.h5

The embeddings will be written into the HDF5 file at `~/experiments/my_experiment/test_embeddings.h5` as dataset `embs`.
#嵌入将作为数据集`embs`写入`〜/ experiments / my_experiment / test_embeddings.h5`的HDF5文件中。
Most relevant settings are automatically loaded from the experiment's `args.json` file, but some can be overruled on the commandline.
#大部分相关的设置都会从实验的`args.json`文件中自动加载，但有些可以在命令行上被忽略。

If the training was performed using data augmentation (highly recommended),
#如果训练是使用数据增强进行的（强烈推荐），可以在嵌入步骤中花更多时间来计算增强嵌入，这通常更加健壮并且在下游任务中表现更好。
one can invest a some more time in the embedding step in order to compute augmented embeddings,
which are usually more robust and perform better in downstream tasks.

The following is an example that computes extensively augmented embeddings:
#以下是计算广泛增强的嵌入的示例：
```
python embed.py \
    --experiment_root ~/experiments/my_experiment \
    --dataset data/market1501_query.csv \
    --filename test_embeddings_augmented.h5 \
    --flip_augment \
    --crop_augment five \
    --aggregator mean
```

This will take 10 times longer, because we perform a total of 10 augmentations per image (2 flips times 5 crops).
#这将花费10倍的时间，因为我们每个图像总共执行10次增强（2次翻转5次）。
All individual embeddings will also be stored in the `.h5` file, thus the disk-space also increases.
#所有单独的嵌入也将被存储在`.h5`文件中，因此磁盘空间也会增加。
One question is how the embeddings of the various augmentations should be combined.
#一个问题是如何组合各种增强的嵌入。
When training using the euclidean metric in the loss, simply taking the mean is what makes most sense,
and also what the above invocation does through `--aggregator mean`.
#当在损失中使用欧氏度量进行训练时，简单地考虑平均值是最有意义的，这也就是上述通过调用“--aggregator mean”所做的事情。

But if one for example trains a normalized embedding (by using a `_normalize` head for instance),
#但是，如果一个例子训练一个规范化的嵌入（例如使用`_normalize`头部），
The embeddings *must* be re-normalized after averaging, and so one should use `--aggregator normalized_mean`.
#嵌入*必须在平均后重新归一化，所以应该使用`--aggregator normalized_mean`。

The final combined embedding is again stored as `embs` in the `.h5` file, as usual.
#像往常一样，最后的组合嵌入再次以'embs`存储在`.h5`文件中。
# Evaluating embeddings  评估嵌入

Once the embeddings have been generated, it is a good idea to compute CMC curves and mAP for evaluation.
#一旦生成嵌入，计算CMC曲线和mAP以进行评估是一个不错的主意。
With only minor modifications, the embedding `.h5` files can be used in
[the official Market1501 MATLAB evaluation code](https://github.com/zhunzhong07/IDE-baseline-Market-1501),
#只需稍作修改，就可以使用嵌入的`.h5`文件[官方Market1501 MATLAB评估代码]，这正是我们为这篇论文所做的。
which is exactly what we did for the paper.

For convenience, and to spite MATLAB, we also implemented our own evaluation code in Python.
#为了方便起见，为了避免使用MATLAB，我们还使用Python实现了自己的评估代码。
This code additionally depends on [scikit-learn](http://scikit-learn.org/stable/),
#此代码还取决于[scikit-learn]（http://scikit-learn.org/stable/），并且仍然只使用TensorFlow来重复使用与训练代码相同的度量实现，以保持一致性。
and still uses TensorFlow only for re-using the same metric implementation as the training code, for consistency.
We verified that it produces the exact same results as the reference implementation.
#我们验证它产生与参考实现方法完全相同的结果。
The following is an example of evaluating a Market1501 model, notice it takes a lot of parameters :smile::
#以下是评估Market1501模型的示例，注意它需要很多参数
```
./evaluate.py \
    --excluder market1501 \
    --query_dataset data/market1501_query.csv \
    --query_embeddings ~/experiments/my_experiment/market1501_query_embeddings.h5 \
    --gallery_dataset data/market1501_test.csv \
    --gallery_embeddings ~/experiments/my_experiment/market1501_test_embeddings.h5 \
    --metric euclidean \
    --filename ~/experiments/my_experiment/market1501_evaluation.json
```
python evaluate.py 
--excluder market1501 
--query_dataset data/market1501_query.csv 
--query_embeddings checkpoint/market1501_query_embeddings.h5 
--gallery_dataset data/market1501_test.csv 
--gallery_embeddings checkpoint/market1501_test_embeddings.h5 
--metric euclidean 
--filename checkpoint/market1501_evaluation.json

The only thing that really needs explaining here is the `excluder`.
#这里真正需要解释的唯一事情就是`excluder'。
For some datasets, especially multi-camera ones,
one often excludes pictures of the query person from the gallery (for that one person)
if it is taken from the same camera.
#对于某些数据集，尤其是多相机数据集，如果是从同一台相机拍摄的，则通常会将该查询人的照片从该相册中排除（针对该一个人）。
#这样，人们就可以获得更多的跨摄像机性能。
This way, one gets more of a feeling for across-camera performance.
Additionally, the Market1501 dataset contains some "junk" images in the gallery which should be ignored too.
All this is taken care of by `excluders`.
#此外，Market1501数据集在画廊中包含一些“垃圾”图像，应该忽略它。所有这些都由“excluders`”来处理。
We provide one for the Market1501 dataset, and a `diagonal` one, which should be used where there is no such restriction,
for example the Stanford Online Products dataset.
#我们提供一个Market1501数据集，并提供一个“对角线”数据集，在没有此限制的情况下使用，例如斯坦福的在线产品数据集。

# :exclamation: Important evaluation NOTE :exclamation:  重要评估注意

The implementation of `mAP` computation has [changed from sklearn v0.18 to v0.19](http://scikit-learn.org/stable/whats_new.html#version-0-19).
#“mAP”计算的实现[已从sklearn v0.18更改为v0.19]（http://scikit-learn.org/stable/whats_new.html#version-0-19）。
The implementation in v0.18 and earlier is exactly the same as in the official Market1501 MATLAB evaluation code, but is [wrong]
(https://github.com/scikit-learn/scikit-learn/pull/7356).
#v0.18及更早版本的实现与官方Market1501 MATLAB评估代码完全相同，但[错误]
The implementation in v0.19 and later leads to a roughly one percentage point increase in `mAP` score.
#在v0.19和更高版本中的实施导致“mAP”分数增加大约1个百分点。
It is not correct to compare values across versions, and again, all values in our paper were computed by the official Market1501 MATLAB code.
#比较不同版本的值是不正确的，我们论文中的所有值都是由官方的Market1501 MATLAB代码计算得出的。

The evaluation code in this repository simply uses the scikit-learn code, and thus **the score depends on which version of scikit-learn you are using**.
#此存储库中的评估代码仅使用scikit-learn代码，因此**分数取决于您使用的scikit-learn的版本**。
Unfortunately, almost no paper mentions which code-base they used and how they computed `mAP` scores, so comparison is difficult.
#不幸的是，几乎没有文章提到他们使用的代码库以及他们如何计算“mAP”得分，因此比较是困难的。
Other frameworks have [the same problem](https://github.com/Cysu/open-reid/issues/50), but we expect many not to be aware of this.
#其他框架也有[同样的问题]（https://github.com/Cysu/open-reid/issues/50），但我们希望很多人不知道这一点。

# Independent re-implementations  独立的重新实现
 
These are the independent re-implementations of our paper that we are aware of,
please send a pull-request to add more:
#这些是我们意识到的我们论文的独立重新实现，请发送合并请求以添加更多内容：
- [Open-ReID](https://github.com/Cysu/open-reid) (PyTorch, MIT license)
- https://github.com/huanghoujing/person-reid-triplet-loss-baseline (PyTorch, no license)
