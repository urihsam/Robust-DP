from dependency import *

def fgm(model_prediction, x, y, eps=0.01, iters=1, sign=True, targeted=True, clip_min=0., clip_max=255.):
    """
    Fast gradient method.
    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).
    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param y: The label palceholder.
    :param eps: The scale factor for noise.
    :param iters: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)
    loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, iters)

    def _body(xadv, i):
        logits, ybar = model_prediction(xadv, use_summary=False)
        loss = loss_fn(labels=y, logits=logits)
        dy_dx, = tf.gradients(loss, xadv) # gradient
        if targeted:
            xadv = tf.stop_gradient(xadv - eps*noise_fn(dy_dx)) # gradient descent -- minimize, targeted label
        else:
            xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx)) # gradient ascent -- maximize, true label
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False)
    return tf.identity(xadv)