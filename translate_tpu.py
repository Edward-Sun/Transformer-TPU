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
Translation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six

import numpy as np
import tensorflow as tf

import models.modeling as modeling
import models.beamsearch as beamsearch


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "nmt_config_file", None,
    "The config json file corresponding to the NMT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "source_input_file", None,
    "Source input tokenized file.")

flags.DEFINE_string(
    "target_output_file", None,
    "Target output tokenized file.")

flags.DEFINE_string(
    "vocab_file", None,
    "Vocabulary file.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_float("decode_alpha", 1.0, "Used by Beam Search.")

flags.DEFINE_integer("decode_length", 20, "Used by Beam Search.")

flags.DEFINE_integer("beam_size", 4, "Used by Beam Search.")

flags.DEFINE_integer("decode_batch_size", 32, "Total test size for training.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def sort_input_file(filename, reverse=True):
  # Read file
  with tf.gfile.Open(filename) as fd:
    inputs = [line.strip() for line in fd]

  input_lens = [
    (i, len(line.strip().split())) for i, line in enumerate(inputs)
  ]

  sorted_input_lens = sorted(input_lens,
                             key=lambda x: x[1],
                             reverse=reverse)
  sorted_keys = {}
  sorted_inputs = []

  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i

  return sorted_keys, sorted_inputs


def get_inference_input(inputs,
                        vocabulary,
                        decode_batch_size=32,
                        num_cpu_threads=2, 
                        eos="<EOS>", 
                        unkId=1):
  dataset = tf.data.Dataset.from_tensor_slices(
    tf.constant(inputs)
  )

  # Split string
  dataset = dataset.map(lambda x: tf.string_split([x]).values,
              num_parallel_calls=num_cpu_threads)

  # Append <EOS>
  dataset = dataset.map(
    lambda x: tf.concat([x, [tf.constant(eos)]], axis=0),
    num_parallel_calls=num_cpu_threads
  )

  # Convert tuple to dictionary
  dataset = dataset.map(
    lambda x: {"source": x, "source_length": tf.shape(x)[0]},
    num_parallel_calls=num_cpu_threads
  )

  dataset = dataset.padded_batch(
    decode_batch_size,
    {"source": [tf.Dimension(None)], "source_length": []},
    {"source": eos, "source_length": 0}
  )

  iterator = dataset.make_one_shot_iterator()
  features = iterator.get_next()

  src_table = tf.contrib.lookup.index_table_from_tensor(
    tf.constant(vocabulary),
    default_value=unkId
  )
  features["source"] = src_table.lookup(features["source"])

  return features


def make_vocab(vocab_file):
  vocab = []

  #insert end-of-sentence/unknown/begin-of-sentence symbols
  vocab.append("<EOS>")
  vocab.append("<UNK>")
  vocab.append("<BOS>")

  with tf.gfile.Open(vocab_file, "r") as fin:
    lines = fin.readlines()

  for line in lines:
    word, freq = line.strip().split()
    if int(freq) >= 5:
      vocab.append(word)
      
  tf.logging.info("Vocabulary size: %d" % (len(vocab)))
      
  return vocab


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  vocabulary = make_vocab(FLAGS.vocab_file)
  
  nmt_config = modeling.NmtConfig.from_json_file(FLAGS.nmt_config_file, 
                                                 vocab_size=len(vocabulary))
 
  vocabulary[0] = nmt_config.eos.encode()
  vocabulary[1] = nmt_config.unk.encode()
  vocabulary[2] = nmt_config.bos.encode()

  # Build Graph
  with tf.Graph().as_default():
    # Read input file
    sorted_keys, sorted_inputs = sort_input_file(FLAGS.source_input_file)
    # Build input queue
    features = get_inference_input(inputs=sorted_inputs, 
                                   vocabulary=vocabulary,
                                   decode_batch_size=FLAGS.decode_batch_size,
                                   eos=nmt_config.eos.encode(),
                                   unkId=nmt_config.unkId)

    # Create placeholders
    placeholders = {
      "source": tf.placeholder(tf.int32, [None, None], "source_0"),
      "source_length": tf.placeholder(tf.int32, [None], "source_length_0")
    }

    model = modeling.NmtModel(config=nmt_config)
    
    model_fn = model.get_inference_func()
    
    predictions = beamsearch.create_inference_graph(model_fns=model_fn, 
                                                    features=placeholders, 
                                                    decode_length=FLAGS.decode_length, 
                                                    beam_size=FLAGS.beam_size,
                                                    top_beams=1, 
                                                    decode_alpha=FLAGS.decode_alpha, 
                                                    bosId=nmt_config.bosId, 
                                                    eosId=nmt_config.eosId)

    # Create assign ops
    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if FLAGS.init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
      tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    total_size = 0
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
      total_size += reduce(lambda x, y: x*y, var.get_shape().as_list())
    tf.logging.info("  total variable parameters: %d", total_size)

    results = []

    # Create session
    tpu_cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project).get_master()

    with tf.Session(tpu_cluster) as sess:
      sess.run(tf.contrib.tpu.initialize_system())
      # Restore variables
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())

      while True:
        try:
          feats = sess.run(features)
          ops = (predictions[0],predictions[1])
          feed_dict = {}
          for name in feats:
            feed_dict[placeholders[name]] = feats[name]
          results.append(sess.run(ops, feed_dict=feed_dict))
          message = "Finished batch %d" % len(results)
          tf.logging.log(tf.logging.INFO, message)
        except tf.errors.OutOfRangeError:
          break
      sess.run(tf.contrib.tpu.shutdown_system())

    # Convert to plain text
    outputs = []

    for result in results:
      for item in result[0]:
        for subitem in item.tolist():
          outputs.append(subitem)

    restored_outputs = []
    for index in range(len(sorted_inputs)):
      restored_outputs.append(outputs[sorted_keys[index]])

    # Write to file
    with tf.gfile.Open(FLAGS.target_output_file, "w") as outfile:
      for output in restored_outputs:
        decoded = []
        for idx in output:
          if isinstance(idx, six.integer_types):
            symbol = vocabulary[idx]
          else:
            symbol = idx

          if symbol == nmt_config.eos.encode():
            break
          decoded.append(symbol)

        decoded = str.join(" ", decoded)

        decoded = decoded.replace("@@ ", "")
        
        outfile.write("%s\n" % decoded)


if __name__ == "__main__":
  flags.mark_flag_as_required("source_input_file")
  flags.mark_flag_as_required("target_output_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("nmt_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  tf.app.run()