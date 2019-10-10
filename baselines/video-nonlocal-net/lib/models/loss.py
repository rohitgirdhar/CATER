from core.config import config as cfg
from caffe2.python import core


def add_losses(model, split, blob_out, labels):
    if split == 'train':
        scale = 1. / cfg.NUM_GPUS
        if cfg.MODEL.MULTILABEL:
            labels = model.Cast(labels, "labels_float", to=core.DataType.FLOAT)
            loss_per_batch = model.SigmoidCrossEntropyWithLogits(
                [blob_out, labels], ['loss_per_batch'])
            loss = model.Scale(
                model.ReduceBackMean(loss_per_batch, 'loss_unscaled'),
                'loss', scale=scale)
            activation = model.Sigmoid(blob_out, 'sigmoid')
        else:
            activation, loss = model.SoftmaxWithLoss(
                [blob_out, labels], ['softmax', 'loss'], scale=scale)
    elif split == 'val':  # in ['test', 'val']:
        if cfg.MODEL.MULTILABEL:
            activation = model.Sigmoid(blob_out, 'sigmoid', engine='CUDNN')
        else:
            activation = model.Softmax(blob_out, 'softmax', engine='CUDNN')
        loss = None
    elif split == 'test':
        # fully convolutional testing
        blob_out = model.Transpose(blob_out, 'pred_tr', axes=(0, 2, 3, 4, 1,))
        blob_out, old_shape = model.Reshape(
            blob_out, ['pred_re', 'pred_shape5d'],
            shape=(-1, cfg.MODEL.NUM_CLASSES))
        if cfg.MODEL.MULTILABEL:
            blob_out = model.Sigmoid(blob_out, 'sigmoid_conv', engine='CUDNN')
            final_name = 'sigmoid'
        else:
            blob_out = model.Softmax(blob_out, 'softmax_conv', engine='CUDNN')
            final_name = 'softmax'
        blob_out = model.Reshape(
            [blob_out, 'pred_shape5d'],
            [final_name + '_conv_re', 'pred_shape2d'])[0]
        blob_out = model.Transpose(
            blob_out, final_name + '_conv_tr', axes=(0, 4, 1, 2, 3))
        blob_out = model.net.ReduceBackMean(
            [blob_out], [final_name + '_ave_w'])
        blob_out = model.ReduceBackMean(
            [blob_out], [final_name + '_ave_h'])
        activation = model.ReduceBackMean(
            [blob_out], [final_name])
        loss = None
    # To have a common handle for the sigmoid/softmax outputs, used in eval
    model.Alias(activation, 'activation')
    return activation, loss
