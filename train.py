#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import timedelta
from importlib import import_module
import logging.config
import os
from signal import SIGINT, SIGTERM
import sys
import time

import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import common
import lbtoolbox as lb
import loss
from nets import NET_CHOICES
from heads import HEAD_CHOICES


parser = ArgumentParser(description='Train a ReID network.')

# Required.

parser.add_argument(
    '--experiment_root', required=True, type=common.writeable_directory,
    help='Location used to store checkpoints and dumped data.')
#'--experiment_root'用于存储checkpoints和转储数据的位置。

parser.add_argument(
    '--train_set',
    help='Path to the train_set csv file.')
#'--train_set',train_set csv文件的路径

parser.add_argument(
    '--image_root', type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')
#'--image_root',在train_set csv中将被预先设置为文件名的路径。

# Optional with sane defaults.  可选的理智默认值。

parser.add_argument(
    '--resume', action='store_true', default=False,
    help='When this flag is provided, all other arguments apart from the '
         'experiment_root are ignored and a previously saved set of arguments '
         'is loaded.')
#'--resume',当提供此标志时，除了experiment_root之外的所有其他参数都将被忽略，并加载之前保存的一组参数。

parser.add_argument(
    '--model_name', default='resnet_v1_50', choices=NET_CHOICES,
    help='Name of the model to use.')
#'--model_name',要使用的模型的名称。

parser.add_argument(
    '--head_name', default='fc1024', choices=HEAD_CHOICES,
    help='Name of the head to use.')
#'--head_name',要使用的head的名称。

parser.add_argument(
    '--embedding_dim', default=128, type=common.positive_int,
    help='Dimensionality of the embedding space.')
#--embedding_dim,嵌入空间的维数


parser.add_argument(
    '--initial_checkpoint', default=None,
    help='Path to the checkpoint file of the pretrained network.')
#'--initial_checkpoint',预训练网络的checkpoint文件的路径。

# TODO move these defaults to the .sh script?
parser.add_argument(
    '--batch_p', default=32, type=common.positive_int,
    help='The number P used in the PK-batches')
#'--batch_p',PK批次中使用的数字P.

parser.add_argument(
    '--batch_k', default=4, type=common.positive_int,
    help='The numberK used in the PK-batches')

parser.add_argument(
    '--net_input_height', default=256, type=common.positive_int,
    help='Height of the input directly fed into the network.')
#'--net_input_height',输入的高度直接馈入网络。

parser.add_argument(
    '--net_input_width', default=128, type=common.positive_int,
    help='Width of the input directly fed into the network.')

parser.add_argument(
    '--pre_crop_height', default=288, type=common.positive_int,
    help='Height used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')
#'--pre_crop_height',用于调整已加载图像大小的高度。没有应用裁剪时，这会被忽略。

parser.add_argument(
    '--pre_crop_width', default=144, type=common.positive_int,
    help='Width used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')
#--pre_crop_width,
# TODO end

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel loading.')
#'--loading_threads',用于并行加载的线程数。

parser.add_argument(
    '--margin', default='soft', type=common.float_or_string,
    help='What margin to use: a float value for hard-margin, "soft" for '
         'soft-margin, or no margin if "none".')
#'--margin',选择margin的模式

parser.add_argument(
    '--metric', default='euclidean', choices=loss.cdist.supported_metrics,
    help='Which metric to use for the distance between embeddings.')
#'--metric'，用于嵌入之间距离的度量标准。

parser.add_argument(
    '--loss', default='batch_hard', choices=loss.LOSS_CHOICES.keys(),
    help='Enable the super-mega-advanced top-secret sampling stabilizer.')
#'--loss'，

parser.add_argument(
    '--learning_rate', default=3e-4, type=common.positive_float,
    help='The initial value of the learning-rate, before it kicks in.')
#'--learning_rate'，学习率的初始值，在它开始之前。

parser.add_argument(
    '--train_iterations', default=25000, type=common.positive_int,
    help='Number of training iterations.')
#'--train_iterations'，训练迭代次数。

parser.add_argument(
    '--decay_start_iteration', default=15000, type=int,
    help='At which iteration the learning-rate decay should kick-in.'
         'Set to -1 to disable decay completely.')
#'--decay_start_iteration'，学习速率衰减应在哪一次迭代中进行。设置为-1可完全禁用衰减。

parser.add_argument(
    '--checkpoint_frequency', default=1000, type=common.nonnegative_int,
    help='After how many iterations a checkpoint is stored. Set this to 0 to '
         'disable intermediate storing. This will result in only one final '
         'checkpoint.')
#'--checkpoint_frequency',在多少次迭代之后，存储检查点。将其设置为0可禁用中间存储。这将导致只有一个最终检查点。

parser.add_argument(
    '--flip_augment', action='store_true', default=False,
    help='When this flag is provided, flip augmentation is performed.')
#'--flip_augment'，当提供此标志时，执行翻转增强。

parser.add_argument(
    '--crop_augment', action='store_true', default=False,
    help='When this flag is provided, crop augmentation is performed. Based on'
         'The `crop_height` and `crop_width` parameters. Changing this flag '
         'thus likely changes the network input size!')
#'--crop_augment'，当提供此标志时，执行裁剪增强。基于`crop_height`和`crop_width`参数。更改此标志可能会改变网络输入大小！

parser.add_argument(
    '--detailed_logs', action='store_true', default=False,
    help='Store very detailed logs of the training in addition to TensorBoard'
         ' summaries. These are mem-mapped numpy files containing the'
         ' embeddings, losses and FIDs seen in each batch during training.'
         ' Everything can be re-constructed and analyzed that way.')
#'--detailed_logs'，“除TensorBoard摘要外，还存储非常详细的培训日志。
#这些是mem-mapped numpy文件，其中包含培训期间在每批中看到的嵌入，损失和FID。一切都可以重新构建和分析。“

def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. 给定一个PID，选择该特定PID的K个FID。"""
    possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))

    # The following simply uses a subset of K of the possible FIDs 如果可用的FID超过或完全可用，下面简单地使用可能FID的K子集。
    # if more than, or exactly K are available. Otherwise, we first 否则，我们首先创建一个填充的索引列表，
    # create a padded list of indices which contain a multiple of the   其中包含原始FID计数的倍数，
    # original FID count such that all of them will be sampled equally likely.这样所有这些索引都将被同等采样。
    count = tf.shape(possible_fids)[0]
    padded_count = tf.cast(tf.ceil(batch_k / tf.cast(count, tf.float32)), tf.int32) * count
    full_range = tf.mod(tf.range(padded_count), count)

    # Sampling is always performed by shuffling and taking the first k.
    shuffled = tf.random_shuffle(full_range)
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k])

    return selected_fids, tf.fill([batch_k], pid)


def main():
    args = parser.parse_args()

    # We store all arguments in a json file. This has two advantages:       我们将所有参数存储在json文件中。 这有两个好处：
    # 1. We can always get back and see what exactly that experiment was    1.我们总是可以回头看看实验是什么
    # 2. We can resume an experiment as-is without needing to remember all flags.2.我们可以按原样恢复实验，无需记住所有标志。
    args_file = os.path.join(args.experiment_root, 'args.json')
    if args.resume:
        if not os.path.isfile(args_file):
            raise IOError('`args.json` not found in {}'.format(args_file))

        print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)
        args_resumed['resume'] = True  # This would be overwritten.

        # When resuming, we not only want to populate the args object with the
        # values from the file, but we also want to check for some possible
        # conflicts between loaded and given arguments.
        #恢复时，我们不仅需要使用文件中的值填充args对象，但我们也想检查加载参数和给定参数之间的一些可能的冲突。
        for key, value in args.__dict__.items():
            if key in args_resumed:
                resumed_value = args_resumed[key]
                if resumed_value != value:
                    print('Warning: For the argument `{}` we are using the'
                          ' loaded value `{}`. The provided value was `{}`'
                          '.'.format(key, resumed_value, value))
                    args.__dict__[key] = resumed_value
            else:
                print('Warning: A new argument was added since the last run:'
                      ' `{}`. Using the new value: `{}`.'.format(key, value))

    else:
        # If the experiment directory exists already, we bail in fear.如果实验目录已经存在，我们就会担心。
        if os.path.exists(args.experiment_root):
            if os.listdir(args.experiment_root):
                print('The directory {} already exists and is not empty.'
                      ' If you want to resume training, append --resume to'
                      ' your call.'.format(args.experiment_root))
                exit(1)
        else:
            os.makedirs(args.experiment_root)

        # Store the passed arguments for later resuming and grepping in a nice
        # and readable format.将传递的参数存储起来，以便稍后以一种很好且可读的格式恢复和刷新。
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    log_file = os.path.join(args.experiment_root, "train")
    logging.config.dictConfig(common.get_logging_dict(log_file))
    log = logging.getLogger('train')

    # Also show all parameter values at the start, for ease of reading logs.
    #同时在开始时显示所有参数值，便于读取日志。
    log.info('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        log.info('{}: {}'.format(key, value))

    # Check them here, so they are not required when --resume-ing.在这里检查他们，
    if not args.train_set:
        parser.print_help()
        log.error("You did not specify the `train_set` argument!")
        sys.exit(1)
    if not args.image_root:
        parser.print_help()
        log.error("You did not specify the required `image_root` argument!")
        sys.exit(1)

    # Load the data from the CSV file.                               加载CSV文件中的数据。
    pids, fids = common.load_dataset(args.train_set, args.image_root)
    max_fid_len = max(map(len, fids))  # We'll need this later for logfiles.我们稍后需要这个日志文件。

    # Setup a tf.Dataset where one "epoch" loops over all PIDS. 设置一个tf.Dataset，其中一个“epoch”在所有PIDS上循环。
    # PIDS are shuffled after every epoch and continue indefinitely.PIDS在每个时代之后都会被洗牌并无限期地继续下去。

    unique_pids = np.unique(pids)
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    dataset = dataset.shuffle(len(unique_pids))

    # Constrain the dataset size to a multiple of the batch-size, so that
    # we don't get overlap at the end of each epoch.将数据集大小限制为批量的倍数，以便在每个时期结束时我们不会重叠。
    dataset = dataset.take((len(unique_pids) // args.batch_p) * args.batch_p)
    dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.  永远重复。 说明它的有趣方式。

    # For every PID, get K images.对于每个PID，获得K个图像。
    dataset = dataset.map(lambda pid: sample_k_fids_for_pid(
        pid, all_fids=fids, all_pids=pids, batch_k=args.batch_k))

    # Ungroup/flatten the batches for easy loading of the files.取消组合/拼合批次以便轻松加载文件。
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # Convert filenames to actual image tensors.                 将文件名转换为实际图像张量。
    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)
    dataset = dataset.map(
        lambda fid, pid: common.fid_to_image(
            fid, pid, image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)

    # Augment the data if specified by the arguments.            如果由参数指定，则增加数据。
    if args.flip_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.image.random_flip_left_right(im), fid, pid))
    if args.crop_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.random_crop(im, net_input_size + (3,)), fid, pid))

    # Group it back into PK batches.                将其重新分组为PK批次。
    batch_size = args.batch_p * args.batch_k
    dataset = dataset.batch(batch_size)

    # Overlap producing and consuming for parallelism.
    dataset = dataset.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator.由于我们无限地重复数据，因此我们只需要一个one-shot 迭代器。
    images, fids, pids = dataset.make_one_shot_iterator().get_next()

    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)

    # Feed the image through the model. The returned `body_prefix` will be used
    # further down to load the pre-trained weights for all variables with this
    # prefix.
    #通过模型提供图像。返回的`body_prefix`将进一步用于加载具有此前缀的所有变量的预先训练权重。
    endpoints, body_prefix = model.endpoints(images, is_training=True)
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=True)

    # Create the loss in two steps:                                       分两步创建损失：
    # 1. Compute all pairwise distances according to the specified metric.1.根据指定的度量计算所有成对距离。
    # 2. For each anchor along the first dimension, compute its loss.     2.对于第一维中的每个锚，计算其损失。
    dists = loss.cdist(endpoints['emb'], endpoints['emb'], metric=args.metric)
    losses, train_top1, prec_at_k, _, neg_dists, pos_dists = loss.LOSS_CHOICES[args.loss](
        dists, pids, args.margin, batch_precision_at_k=args.batch_k-1)

    # Count the number of active entries, and compute the total batch loss.  计算活动条目的数量，并计算总批量损失。
    num_active = tf.reduce_sum(tf.cast(tf.greater(losses, 1e-5), tf.float32))
    loss_mean = tf.reduce_mean(losses)

    # Some logging for tensorboard.            一些日志记录在tensorboard。
    tf.summary.histogram('loss_distribution', losses)
    tf.summary.scalar('loss', loss_mean)
    tf.summary.scalar('batch_top1', train_top1)
    tf.summary.scalar('batch_prec_at_{}'.format(args.batch_k-1), prec_at_k)
    tf.summary.scalar('active_count', num_active)
    tf.summary.histogram('embedding_dists', dists)
    tf.summary.histogram('embedding_pos_dists', pos_dists)
    tf.summary.histogram('embedding_neg_dists', neg_dists)
    tf.summary.histogram('embedding_lengths',
                         tf.norm(endpoints['emb_raw'], axis=1))

    # Create the mem-mapped arrays in which we'll log all training detail in
    # addition to tensorboard, because tensorboard is annoying for detailed
    # inspection and actually discards data in histogram summaries.
    #创建mem - mapped数组，我们将记录除tensorboard之外的所有训练细节，因为tensorboard对于详细检查很烦人，实际上在直方图总结中丢弃了数据。

    if args.detailed_logs:
        log_embs = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'embeddings'),
            dtype=np.float32, shape=(args.train_iterations, batch_size, args.embedding_dim))
        log_loss = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'losses'),
            dtype=np.float32, shape=(args.train_iterations, batch_size))
        log_fids = lb.create_or_resize_dat(
            os.path.join(args.experiment_root, 'fids'),
            dtype='S' + str(max_fid_len), shape=(args.train_iterations, batch_size))

    # These are collected here before we add the optimizer, because depending
    # on the optimizer, it might add extra slots, which are also global
    # variables, with the exact same prefix.
    #在我们添加优化器之前收集这些信息，因为根据优化器的不同，它可能会添加额外的插槽，这些插槽也是全局变量，具有完全相同的前缀。
    model_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)

    # Define the optimizer and the learning-rate schedule.              定义优化器和学习率计划。
    # Unfortunately, we get NaNs if we don't handle no-decay separately. 不幸的是，如果我们不单独处理无衰减，我们会得到NaNs。
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if 0 <= args.decay_start_iteration < args.train_iterations:
        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            tf.maximum(0, global_step - args.decay_start_iteration),
            args.train_iterations - args.decay_start_iteration, 0.001)
    else:
        learning_rate = args.learning_rate
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Feel free to try others!                               随意尝试别的的优化器！
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate)

    # Update_ops are used to update batchnorm stats.  Update_ops用于更新batchnorm统计信息。
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss_mean, global_step=global_step)

    # Define a saver for the complete model.  为整个模型定义一个保存器。
    checkpoint_saver = tf.train.Saver(max_to_keep=0)

    with tf.Session() as sess:
        if args.resume:
            # In case we're resuming, simply load the full checkpoint to init.如果我们正在恢复，只需将完整的检查点加载到init。
            last_checkpoint = tf.train.latest_checkpoint(args.experiment_root)
            log.info('Restoring from checkpoint: {}'.format(last_checkpoint))
            checkpoint_saver.restore(sess, last_checkpoint)
        else:
            # But if we're starting from scratch, we may need to load some
            # variables from the pre-trained weights, and random init others.
            #但是如果我们从头开始，我们可能需要从预先训练的权重中加载一些变量，并随机初始化其他的变量。
            sess.run(tf.global_variables_initializer())
            if args.initial_checkpoint is not None:
                saver = tf.train.Saver(model_variables)
                saver.restore(sess, args.initial_checkpoint)

            # In any case, we also store this initialization as a checkpoint,
            # such that we could run exactly reproduceable experiments.
            #无论如何，我们也将这个初始化作为检查点存储，以便我们可以运行完全可再生的实验。
            checkpoint_saver.save(sess, os.path.join(
                args.experiment_root, 'checkpoint'), global_step=0)

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.experiment_root, sess.graph)

        start_step = sess.run(global_step)
        log.info('Starting training from iteration {}.'.format(start_step))

        # Finally, here comes the main-loop. This `Uninterrupt` is a handy
        # utility such that an iteration still finishes on Ctrl+C and we can
        # stop the training cleanly.
        #最后，这里是主循环。这个`Uninterrupt`是一个非常方便的工具，可以在Ctrl + C之后完成迭代才停止，我们可以干净地停止训练。
        with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            for i in range(start_step, args.train_iterations):

                # Compute gradients, update weights, store logs!计算梯度，更新权重，存储日志！
                start_time = time.time()
                _, summary, step, b_prec_at_k, b_embs, b_loss, b_fids = \
                    sess.run([train_op, merged_summary, global_step,
                              prec_at_k, endpoints['emb'], losses, fids])
                elapsed_time = time.time() - start_time

                # Compute the iteration speed and add it to the summary.计算迭代速度并将其添加到摘要中。
                # We did observe some weird spikes that we couldn't track down.我们确实观察到一些我们无法追查的奇怪尖峰。
                summary2 = tf.Summary()
                summary2.value.add(tag='secs_per_iter', simple_value=elapsed_time)
                summary_writer.add_summary(summary2, step)
                summary_writer.add_summary(summary, step)

                if args.detailed_logs:
                    log_embs[i], log_loss[i], log_fids[i] = b_embs, b_loss, b_fids

                # Do a huge print out of the current progress.            在当前的进展中做大量的印刷。
                seconds_todo = (args.train_iterations - step) * elapsed_time
                log.info('iter:{:6d}, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, '
                         'batch-p@{}: {:.2%}, ETA: {} ({:.2f}s/it)'.format(
                             step,
                             float(np.min(b_loss)),
                             float(np.mean(b_loss)),
                             float(np.max(b_loss)),
                             args.batch_k-1, float(b_prec_at_k),
                             timedelta(seconds=int(seconds_todo)),
                             elapsed_time))
                sys.stdout.flush()
                sys.stderr.flush()

                # Save a checkpoint of training every so often.每隔一段时间保存一次训练的检查点checkpoint。
                if (args.checkpoint_frequency > 0 and
                        step % args.checkpoint_frequency == 0):
                    checkpoint_saver.save(sess, os.path.join(
                        args.experiment_root, 'checkpoint'), global_step=step)

                # Stop the main-loop at the end of the step, if requested. 如果需要，在步骤结束时停止主循环。
                if u.interrupted:
                    log.info("Interrupted on request!")
                    break

        # Store one final checkpoint. This might be redundant, but it is crucial
        # in case intermediate storing was disabled and it saves a checkpoint
        # when the process was interrupted.
        #存储一个最终检查点。 这可能是多余的，但是在中间存储被禁用的情况下它是至关重要的，并且在进程被中断时保存检查点。
        checkpoint_saver.save(sess, os.path.join(
            args.experiment_root, 'checkpoint'), global_step=step)


if __name__ == '__main__':
    main()
