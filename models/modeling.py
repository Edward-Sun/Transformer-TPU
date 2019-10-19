# coding=utf-8
# Copyright 2019 The 11731(Transformer-TPU) Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Transformer Network
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf


class NmtConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               eos="<EOS>",
               unk="<UNK>",
               bos="<BOS>",
               eosId=0,
               unkId=1,
               bosId=2,
               hidden_size=512,
               filter_size=2048,
               num_heads=8,
               num_encoder_layers=6,
               num_decoder_layers=6,
               attention_dropout=0.1,
               hidden_dropout=0.1,
               relu_dropout=0.1,
               label_smoothing=0.1,
               attention_key_channels=0,
               attention_value_channels=0,
               shared_embedding_and_softmax_weights=True,
               shared_source_target_embedding=True):
    self.vocab_size = vocab_size
    self.bos = bos
    self.unk = unk
    self.eos = eos
    self.eosId = eosId
    self.unkId = unkId
    self.bosId = bosId
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.num_heads = num_heads
    self.num_encoder_layers = num_encoder_layers
    self.num_decoder_layers = num_decoder_layers
    self.attention_dropout = attention_dropout
    self.hidden_dropout = hidden_dropout
    self.relu_dropout = relu_dropout
    self.label_smoothing = label_smoothing
    self.attention_key_channels = attention_key_channels
    self.attention_value_channels = attention_value_channels
    self.shared_embedding_and_softmax_weights = shared_embedding_and_softmax_weights
    self.shared_source_target_embedding = shared_source_target_embedding

  @classmethod
  def from_dict(cls, json_object, vocab_size):
    """Constructs a `NmtConfig` from a Python dictionary of parameters."""
    config = NmtConfig(vocab_size=vocab_size)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file, vocab_size):
    """Constructs a `NmtConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text), vocab_size)

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)
  
  
class NmtModel(object):
  """Nmt Model: Transformer"""
  def __init__(self, config):
    self._scope = "Trasnformer"
    self._config = config

  def encoder_subgraph(self, features, mode):
    if mode != "train":
      self._config.hidden_dropout = 0.0
      self._config.attention_dropout = 0.0
      self._config.relu_dropout = 0.0
      self._config.label_smoothing = 0.0

    config = self._config
    # src_seq is a list of lengths of each snt in one batch
    src_seq = features["source"]
    src_len = features["source_length"]
    # create mask matrix with src_seq
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    hidden_size = config.hidden_size
    initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

    src_vocab_size = config.vocab_size
    
    # embedding bias for European languages
    src_embedding = tf.get_variable("shared_embedding",
                                    [src_vocab_size, hidden_size],
                                    initializer=initializer)
    bias = tf.get_variable("src_language_bias", 
                           [hidden_size], 
                           initializer=tf.zeros_initializer)

    # id => embedding
    # src_seq: [batch, max_src_length]
    # src_seq has been appended <EOS>
    # input: [batch, max_src_length, hidden_size]
    # src_mask: [batch, max_src_length]
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)
    inputs = add_timing_signal(inputs)

    # Preparing encoder
    # bias is a language related parameter
    encoder_input = tf.nn.bias_add(inputs, bias)

    # enc_attn_bias: [batch, 1, 1, max_src_length]
    enc_attn_bias = attention_mask(src_mask)

    encoder_input = dropout(encoder_input, config.hidden_dropout)

    with tf.variable_scope("encoder"):
      x = layer_norm(encoder_input)
      # n layer encoder 
      # it is the same in both training and decoding
      for layer in range(config.num_encoder_layers):
        with tf.variable_scope("layer_%d" % layer):
          with tf.variable_scope("self_attention"):
            y = multihead_self_attention(
              x,
              enc_attn_bias,
              config.num_heads,
              config.attention_key_channels or config.hidden_size,
              config.attention_value_channels or config.hidden_size,
              config.hidden_size,
              config.attention_dropout
            )
            y = y["outputs"]
            x = post_process(x, y, config.hidden_dropout)

          with tf.variable_scope("feed_forward"):
            y = ffn_layer(
              x,
              config.filter_size,
              config.hidden_size,
              config.relu_dropout
            )
            x = post_process(x, y, config.hidden_dropout)

      encoder_output = x
    
    return encoder_output

  def decoder_subgraph(self, features, state, mode):
    if mode != "train":
      self._config.hidden_dropout = 0.0
      self._config.attention_dropout = 0.0
      self._config.relu_dropout = 0.0
      self._config.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"] # lengths of each sentence in the batch
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    config = self._config
    hidden_size = config.hidden_size
    tgt_vocab_size = config.vocab_size
    initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      tgt_embedding = tf.get_variable("shared_embedding",
                      [tgt_vocab_size, hidden_size],
                      initializer=initializer)  

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      weights = tf.get_variable("shared_embedding",
                      [tgt_vocab_size, hidden_size],
                      initializer=initializer)

    bias = tf.get_variable("output_bias", 
                           [tgt_vocab_size], 
                           initializer=tf.zeros_initializer)

    # id => embedding
    # tgt_seq: [batch, max_tgt_length]
    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)
    targets = targets * tf.expand_dims(tgt_mask, -1)

    # Preparing encoder and decoder input
    enc_attn_bias = attention_mask(src_mask)
    # Preparing decoder mask
    dec_attn_bias = causal_mask(tf.shape(targets)[1])
    # Shift left
    # input embedding for decoder is from word0 to wordn-1 
    # the targets is w1,w2...wn,<\s>
    # tf.pad: append zero vector
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = add_timing_signal(decoder_input)

    decoder_input = dropout(decoder_input, config.hidden_dropout)

    encoder_output = state["encoder"]

    if mode != "infer": #train
      decoder_state = None
    else:
      decoder_state = state["decoder"]
      decoder_input = decoder_input[:, -1:, :]
      dec_attn_bias = dec_attn_bias[:, :, -1:, :]

    with tf.variable_scope("decoder"):
      x = layer_norm(decoder_input)
      # a dict to store the previous QV of each layer
      # next_state: (name, tensor)
      next_state = {} 
      # n-layer decoder
      for layer in range(config.num_decoder_layers):
        layer_name = "layer_%d" % layer
        with tf.variable_scope(layer_name):
          layer_state = decoder_state[layer_name] if decoder_state is not None else None

          with tf.variable_scope("self_attention"):
            y = multihead_self_attention(
              x,
              dec_attn_bias,
              config.num_heads,
              config.attention_key_channels or config.hidden_size,
              config.attention_value_channels or config.hidden_size,
              config.hidden_size,
              config.attention_dropout,
              state=layer_state
            )

            if layer_state is not None:
              next_state[layer_name] = y["state"]

            y = y["outputs"]
            x = post_process(x, y, config.hidden_dropout)

          with tf.variable_scope("encdec_attention"):
            y = multihead_encdec_attention(
              x,
              encoder_output,
              enc_attn_bias,
              config.num_heads,
              config.attention_key_channels or config.hidden_size,
              config.attention_value_channels or config.hidden_size,
              config.hidden_size,
              config.attention_dropout
            )
            x = post_process(x, y, config.hidden_dropout)

          with tf.variable_scope("feed_forward"):
            y = ffn_layer(
              x,
              config.filter_size,
              config.hidden_size,
              config.relu_dropout,
            )
            x = post_process(x, y, config.hidden_dropout)

      decoder_output = x
      decoder_state = next_state

    if mode != "infer": #train
      # print(decoder_output.shape) # this code will print (?,?,512)
      # [batch, length, hidden] => [batch * length, vocab_size]
      decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
      logits = tf.matmul(decoder_output, weights, False, True)
      logits = tf.nn.bias_add(logits, bias)

      # golden target ids
      labels = tf.reshape(features["target"], [-1])
      weights = tf.reshape(tgt_mask, [-1])
      onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=tgt_vocab_size)
      
      xentropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
                                                 logits=logits,
                                                 weights=weights,
                                                 label_smoothing=config.label_smoothing)

      n = tf.to_float(tgt_vocab_size - 1)
      p = 1.0 - config.label_smoothing
      q = config.label_smoothing / n
      normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

      loss = xentropy - normalizing
      return loss
    else:
      decoder_output = decoder_output[:, -1, :]
      logits = tf.matmul(decoder_output, weights, False, True)
      logits = tf.nn.bias_add(logits, bias)
      log_prob = tf.nn.log_softmax(logits)
      
      return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

  def get_training_func(self):
    def training_fn(features, reuse=None):
      with tf.variable_scope(self._scope, reuse=reuse):
        mode = "train"
        encoder_output = self.encoder_subgraph(features, mode)
        state = {
          "encoder": encoder_output
        }
        loss = self.decoder_subgraph(features, state, mode)   
        return loss
    return training_fn

  def get_inference_func(self):
    def encoding_fn(features, config=None):
      with tf.variable_scope(self._scope):
        encoder_output = self.encoder_subgraph(features, "infer")
        batch = tf.shape(encoder_output)[0]

        state = {
          "encoder": encoder_output,
          "decoder": {
            "layer_%d" % i: {
              "key": tf.zeros([batch, 0, self._config.attention_key_channels 
                               or self._config.hidden_size]),
              "value": tf.zeros([batch, 0, self._config.attention_value_channels 
                                 or self._config.hidden_size])
            }
            for i in range(self._config.num_decoder_layers)
          }
        }
      return state

    def decoding_fn(features, state, config=None):
      with tf.variable_scope(self._scope):
        log_prob, new_state = self.decoder_subgraph(features, state, "infer")

      return log_prob, new_state

    return encoding_fn, decoding_fn


def get_shape_list(tensor, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.
  Args:
    tensor: A tf.Tensor object to find the shape of.
    name: Optional name of the tensor for the error message.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def dropout(input_tensor, dropout_prob):
  """Perform dropout.
  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).
  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def linear(input_data, 
           output_size, 
           bias=True,
           name=None):
  """
  output = input_data * W + b
  """
  with tf.variable_scope(name, default_name="linear"):
    input_shape = get_shape_list(input_data)
    input_size = input_shape[-1]
    output_shape = input_shape[:-1] + [output_size]

    weight_initializer = tf.variance_scaling_initializer(1.0,
                                                         mode="fan_avg",
                                                         distribution="uniform")

    W = tf.get_variable("W", 
                        shape=[input_size, output_size], 
                        initializer=weight_initializer)
    output = tf.matmul(tf.reshape(input_data, [-1, input_size]), W)

    if bias:
      bias = tf.get_variable("b", 
                             shape=[output_size],
                             initializer=tf.zeros_initializer)
      output = output + bias
    
    output = tf.reshape(output, output_shape)

    return output


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def post_process(previous_data,
                 input_data,
                 dropout_rate=None):
  input_data = dropout(input_data, dropout_rate)
  return layer_norm(previous_data + input_data)


def ffn_layer(inputs, hidden_size, output_size, dropout_rate=None, name=None):
  hidden = linear(inputs, hidden_size, name="input_layer")
  hidden = tf.nn.relu(hidden)
  hidden = dropout(hidden, dropout_rate)
  output = linear(hidden, output_size, name="output_layer")
  return output


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
  """This function adds a bunch of sinusoids of different frequencies to a Tensor."""
  with tf.name_scope(name, default_name="add_timing_signal"):
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2

    log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1)
    )
    inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
    )

    scaled_time = (tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0))
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])

    return x + signal


def dot_product_attention(q,
                          k,
                          v,
                          bias=None,
                          attention_dropout_rate=None,
                          num_heads=None,
                          name=None):
  """dot-product attention."""
  # split heads
  q_shape = get_shape_list(q)
  k_shape = get_shape_list(k)
  v_shape = get_shape_list(v)

  head_size = q_shape[-1] // num_heads
  value_size = v_shape[-1] // num_heads

  new_q_shape = q_shape[:-1] + [num_heads, head_size]
  new_k_shape = k_shape[:-1] + [num_heads, head_size]
  new_v_shape = v_shape[:-1] + [num_heads, value_size]

  q = tf.transpose(tf.reshape(q, new_q_shape), [0, 2, 1, 3])
  k = tf.transpose(tf.reshape(k, new_k_shape), [0, 2, 1, 3])
  v = tf.transpose(tf.reshape(v, new_v_shape), [0, 2, 1, 3])

  # [batch, num_heads, query_length, memory_length]
  logits = tf.matmul(q, k, transpose_b=True) * (head_size ** -0.5)
  if bias is not None:
    logits += bias
  weights = tf.nn.softmax(logits)
  weights = dropout(weights, attention_dropout_rate)
  x = tf.matmul(weights, v)

  # combine heads
  new_x_shape = q_shape[:-1] + [v_shape[-1]]
  x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), new_x_shape)
  return x


def attention_mask(mask, neg_inf=-1e9, name=None):
  with tf.name_scope(name, default_name="attention_mask"):
    ret = (1.0 - mask) * neg_inf
    return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def causal_mask(length, neg_inf=-1e9, name=None):
  with tf.name_scope(name, default_name="causal_mask"):
    lower_triangle = tf.matrix_band_part(
      tf.ones([length, length]), -1, 0
    )
    ret = neg_inf * (1.0 - lower_triangle)
    return tf.reshape(ret, [1, 1, length, length])


def multihead_self_attention(queries,
                             bias, 
                             num_heads, 
                             key_size,
                             value_size, 
                             output_size,
                             dropout_rate=None,
                             state=None):
  """Multi-head self attention with input/output transformations."""
  # self attention
  q = linear(queries, key_size, name="q_transform")
  k = linear(queries, key_size, name="k_transform")
  v = linear(queries, value_size, name="v_transform")

  if state is not None:
    # incrementally append current KV to previous KV
    k = tf.concat([state["key"], k], axis=1)
    v = tf.concat([state["value"], v], axis=1)
    next_state = {}
    next_state["key"] = k
    next_state["value"] = v

  results = dot_product_attention(q, k, v, bias, dropout_rate, num_heads)

  outputs = linear(results, output_size, name="output_transform")

  outputs = {"outputs": outputs}
  if state is not None:
    outputs["state"] = next_state

  return outputs


def multihead_encdec_attention(queries,
                               memories, 
                               bias, 
                               num_heads, 
                               key_size,
                               value_size, 
                               output_size,
                               dropout_rate=None):
  """Multi-head encdec attention with input/output transformations."""
  q = linear(queries, key_size, name="q_transform")
  k = linear(memories, key_size, name="k_transform")
  v = linear(memories, value_size, name="v_transform")

  results = dot_product_attention(q, k, v, bias, dropout_rate, num_heads)

  outputs = linear(results, output_size, name="output_transform")

  return outputs