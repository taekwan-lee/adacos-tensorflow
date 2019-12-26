import tensorflow as tf
import math


def calculate_adacos_logits(embds, labels, one_hot, embedding_size, class_num,
                            is_dynamic=True):
  weights = tf.get_variable(name='final_dense',
                            shape=[embedding_size, class_num],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            trainable=True)
  init_s = math.sqrt(2) * math.log(class_num - 1)
  adacos_s = tf.get_variable(name='adacos_s_value', dtype=tf.float32,
                             initializer=tf.constant(init_s),
                             trainable=False,
                             aggregation=tf.VariableAggregation.MEAN)
  embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
  weights = tf.nn.l2_normalize(weights, axis=0)

  logits_before_s = tf.matmul(embds, weights, name='adacos_logits_before_s')

  if is_dynamic == False:
    output = tf.multiply(init_s, logits_before_s, name='adacos_fixed_logits')
    return output

  theta = tf.acos(tf.clip_by_value(logits_before_s, -1.0 + 1e-10, 1.0 - 1e-10))
    
  B_avg = tf.where_v2(tf.less(one_hot, 1),
                      tf.exp(adacos_s*logits_before_s), tf.zeros_like(logits_before_s))
  B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1), name='B_avg')
  #B_avg = tf.stop_gradient(B_avg)
  idxs = tf.squeeze(labels)
  theta_class = tf.gather_nd(theta, tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1),
                             name='theta_class')
  theta_med = tf.contrib.distributions.percentile(theta_class, q=50)
  #theta_med = tf.stop_gradient(theta_med)
  
  with tf.control_dependencies([theta_med, B_avg]):
    temp_s = tf.log(B_avg) / tf.cos(tf.minimum(math.pi/4, theta_med))
    adacos_s = tf.assign(adacos_s, temp_s)
    output = tf.multiply(adacos_s, logits_before_s, name='adacos_dynamic_logits')
    
  return output
