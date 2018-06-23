import numbers
import tensorflow as tf


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.#返回a - b的所有组合的张量。

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        a（2D张量）：一批形状为（B1，F）的矢量。
        b (2D tensor): A batch of vectors shaped (B2, F).
        b（2D张量）：一批形状为（B2，F）的矢量。

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
        在'a'和'b'中的所有向量之间的所有成对差异的矩阵将具有形状（B1，B2）。

    Note:
        For convenience, if either `a` or `b` is a `Distribution` object, its
        mean is used.
        为方便起见，如果`a`或`b`是`Distribution`对象，则使用其平均值。
    """
    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)
    #tf.expand_dims(input, axis=None, name=None, dim=None)在第axis位置增加一个维度，
    #[B1,1,F]-[1,B2,F] = [B1,B2]


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.与scipy.spatial的cdist类似，但具有象征意义

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
    目前支持的指标可以列为“cdist.supported_metrics”，它们是：
        - 'euclidean', although with a fudge-factor epsilon.
         'euclidean欧几里得距离'，尽管具有附加因子epsilon ε。
        - 'sqeuclidean', the squared euclidean.
          欧几里得距离的平方
        - 'cityblock', the manhattan or L1 distance.
        曼哈顿距离或者L1

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        一个（二维张量）：左侧，形状（B1，F）。
        b (2D tensor): The right-hand side, shaped (B2, F).
        b（二维张量）：右侧，形状（B2，F）。
        metric (string): Which distance metric to use, see notes.
        指标（字符串）：使用哪个距离度量，请参阅说明。

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).在'a'和'b'中的所有向量之间的所有成对距离的矩阵将具有形状（B1，B2）。

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    当采用平方根时（例如在欧几里得情况下），因为在零处的平方根的梯度是未定义的，所以添加小ε。
    因此，在这些情况下，它永远不会返回精确的零。
    """
    with tf.name_scope("cdist"):
        diffs = all_diffs(a, b)
        if metric == 'sqeuclidean':
            return tf.reduce_sum(tf.square(diffs), axis=-1)
        elif metric == 'euclidean':
            return tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-12)
        elif metric == 'cityblock':
            return tf.reduce_sum(tf.abs(diffs), axis=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))
        #如果选择的距离不是这三个，报错。.format函数，进行格式转换，以大括号{}为标志。
        #输出The following metric is not implemented by `cdist` yet：metric
cdist.supported_metrics = [
    'euclidean',
    'sqeuclidean',
    'cityblock',
]


def get_at_indices(tensor, indices):
    """ Like `tensor[np.arange(len(tensor)), indices]` in numpy.
        像`tensor[np.arange（len（tensor）），索引]`在numpy。
        tensor的前len(tesnor)行按照tensor进行索引，tensor中的值不能超出indices的列数。
    """
    counter = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
    #indices.dtype输出数据的格式。获得indices的大小
    return tf.gather_nd(tensor, tf.stack((counter, indices), -1))
    #函数：tf.gather_nd(params, indices, name=None)  将参数中的切片收集到由索引indices指定的形状的张量中。
    #这里parama=tensor,indices=tf.stack((counter,indices),-1)

def batch_hard(dists, pids, margin, batch_precision_at_k=None):
    """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

    Args:
        dists (2D tensor): A square all-to-all distance matrix as given by cdist.
        dists（二维张量）：由cdist给出的一个距离矩阵的平方。
        pids (1D tensor): The identities of the entries in `batch`, shape (B,).
        pids（一维张量）：`批量`中身份的标识，形状为（B，）。
            This can be of any type that can be compared, thus also a string.
            这可以是任何可以比较的类型，也可以是字符串。
        margin: The value of the margin if a number, alternatively the string
            'soft' for using the soft-margin formulation, or `None` for not
            using a margin at all.
        余量：如果是数字，则为余量的值，或者使用soft-margin公式的“soft”字符串，或者根本不使用边距的“None”。

    Returns:
        A 1D tensor of shape (B,) containing the loss value for each sample.
        包含每个样本损失值的1D张量形状（B，）。
    """
    with tf.name_scope("batch_hard"):
        same_identity_mask = tf.equal(tf.expand_dims(pids, axis=1),
                                      tf.expand_dims(pids, axis=0))
        #tf.equal(A,B)对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
        #tf.expand_dims(pids, axis=1)形状为（B,1),   tf.expand_dims(pids, axis=0)形状为（1,B)
        negative_mask = tf.logical_not(same_identity_mask)
        #负样本的掩码，逻辑非
        positive_mask = tf.logical_xor(same_identity_mask,
                                       tf.eye(tf.shape(pids)[0], dtype=tf.bool))
        #正样本的掩码，逻辑异或。逐元素地计算x^y的真值表，每个元素的dtype属性必须为tf.bool
        #tf.eye(num_rows,)构造单位矩阵,num_rows矩阵的行数。

        furthest_positive = tf.reduce_max(dists*tf.cast(positive_mask, tf.float32), axis=1)
        #最远的正样本，tf.cast(x, dtype)将x的数据格式转化成dtype.例如，原来x的数据格式是bool， 那么将其转化成float以后，就能够将其转化成0和1的序列。
        #tf.reduce_max(input_tensor,axis=1),计算一个张量的每一行上元素的最大值。

        closest_negative = tf.map_fn(lambda x: tf.reduce_min(tf.boolean_mask(x[0], x[1])),
                                    (dists, negative_mask), tf.float32)
        #最近的负样本，tf.boolean_mask(a,b) 将使a (m维)矩阵仅保留与b中“True”元素同下标的部分，并将结果展开到m-1维。
        #tf.reduce_min,沿着tensor的某一维度，计算元素的最小值。     x[0]=dists,x[1]=negative_mask
        #tf.map_fn(fn,elems,dtype=None),将函数当成参数传入，elems是需要做处理的Tensors，TF将会将elems从第一维展开，进行map处理

        # Another way of achieving the same, though more hacky:
        # closest_negative = tf.reduce_min(dists + 1e5*tf.cast(same_identity_mask, tf.float32), axis=1)

        diff = furthest_positive - closest_negative#最远正样本-最近的负样本
        if isinstance(margin, numbers.Real):    #numbers.Real表示实数类
            #isinstance(object, classinfo),object实例对象,classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
            #如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。
            diff = tf.maximum(diff + margin, 0.0)
        elif margin == 'soft':
            diff = tf.nn.softplus(diff)
        elif margin.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                'The margin {} is not implemented in batch_hard'.format(margin))

    if batch_precision_at_k is None:
        return diff

    # For monitoring, compute the within-batch top-1 accuracy and the为了进行监控，计算批次内的top-1准确度和批量内精确度K，这更具表现力。
    # within-batch precision-at-k, which is somewhat more expressive.
    with tf.name_scope("monitoring"):
        # This is like argsort along the last axis. Add one to K as we'll
        # drop the diagonal.这就像按照最后一个维度进行排序一样。 将1添加到K，因为我们将放弃对角线。
        _, indices = tf.nn.top_k(-dists, k=batch_precision_at_k+1)
        #tf.nn.top_k(input, k, name=None),这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引。

        # Drop the diagonal (distance to self is always least).＃放弃对角线（自我的距离总是最小）。
        indices = indices[:,1:]  #即取所有数据的第1到n-1列数据，去掉最小的

        # Generate the index indexing into the batch dimension.生成批量维度中索引的索引。
        # This is simething like [[0,0,0],[1,1,1],...,[B,B,B]]
        batch_index = tf.tile(
            tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),   #tf.shape(indices)[0] =  B,
            (1, tf.shape(indices)[1]))#  tf.shape(indices)[1] =1   (1, tf.shape(indices)[1])=(1,1)
        #tf.tile(input, multiples, name=None),通过拼接一个给定的tensor构建一个新的tensor,
        #output tensor 的第i维有 input.dims(i) * multiples[i] 个元素
        #multiples: 一维整数tensor， 长度和输入tensor的维度相同,tiling `[a b c d]` by `[2]`produces`[a b c d a b c d]`.

        # Stitch the above together with the argsort indices to get the indices of the top-k of each row.
        #将上面的内容与argsort索引一起拼接，得到每一行的top-k索引。
        topk_indices = tf.stack((batch_index, indices), -1)

        # See if the topk belong to the same person as they should, or not.
        #看看topk是否属于同一个人，或者不是。
        topk_is_same = tf.gather_nd(same_identity_mask, topk_indices)

        # All of the above could be reduced to the simpler following if k==1
        #top1_is_same = get_at_indices(same_identity_mask, top_idxs[:,1])
        #如果k == 1,top1_is_same = get_at_indices（same_identity_mask,top_idxs [：,1]），则上述所有可以简化为更简单的后续

        topk_is_same_f32 = tf.cast(topk_is_same, tf.float32)
        top1 = tf.reduce_mean(topk_is_same_f32[:,0]) #第一列,如果不指定第二个参数，那么就在所有第一列的元素中取平均值
        prec_at_k = tf.reduce_mean(topk_is_same_f32)#如果不指定第二个参数，那么就在所有的元素中取平均值

        # Finally, let's get some more info that can help in debugging while
        # we're at it!
        negative_dists = tf.boolean_mask(dists, negative_mask)
        positive_dists = tf.boolean_mask(dists, positive_mask)
        #tf.boolean_mask(a,b) 将使a (m维)矩阵仅保留与b中“True”元素同下标的部分，并将结果展开到m-1维。

        return diff, top1, prec_at_k, topk_is_same, negative_dists, positive_dists


LOSS_CHOICES = {
    'batch_hard': batch_hard,
}
