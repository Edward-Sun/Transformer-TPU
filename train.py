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
"""NMT"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import models.modeling as modeling
import optimization


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
    "target_input_file", None,
    "Target input tokenized file.")

flags.DEFINE_string(
    "vocab_file", None,
    "Vocabulary file.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How many model checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

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


def model_fn_builder(nmt_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.NmtModel(config=nmt_config)

    loss = model.get_training_func()(features)
    
    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

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

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(source_input_file,
                     target_input_file,
                     vocabulary,
                     max_seq_length,
                     num_cpu_threads=4,
                     eos="<EOS>",
                     unkId=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"] // max_seq_length

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    src_dataset = tf.data.TextLineDataset(source_input_file)
    tgt_dataset = tf.data.TextLineDataset(target_input_file)
    
    dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    # read params.buffer_size sentences for training
    dataset = dataset.shuffle(10000)
    dataset = dataset.repeat()

    dataset = dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values,
            tf.string_split([tgt]).values
        ),
        num_parallel_calls=num_cpu_threads
    )

    dataset = dataset.map(
        lambda src, tgt: (
            tf.concat([src, [tf.constant(eos)]], axis=0),
            tf.concat([tgt, [tf.constant(eos)]], axis=0)
        ),
        num_parallel_calls=num_cpu_threads
    )

    dataset = dataset.map(
        lambda src, tgt: (
          src, 
          tgt,
          tf.shape(src),
          tf.shape(tgt)
        ),
        num_parallel_calls=num_cpu_threads
    )

    dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes=(max_seq_length, max_seq_length, 1, 1),
      drop_remainder=True)

    dataset = dataset.map(
        lambda src, tgt, src_len, tgt_len: {
            "source": src,
            "target": tgt,
            "source_length": src_len,
            "target_length": tgt_len
        },
        num_parallel_calls=num_cpu_threads
    )
      
    # Create lookup table: convert str-dict to id-dict
    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(vocabulary),
        default_value=unkId
    )
    tgt_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(vocabulary),
        default_value=unkId
    )

    dataset = dataset.map(
        lambda feature: _decode_record(feature, src_table, tgt_table),
        num_parallel_calls=num_cpu_threads
    )

    return dataset

  return input_fn


def _decode_record(feature, src_table, tgt_table):
  # String to index lookup
  feature["source"] = src_table.lookup(feature["source"])
  feature["target"] = tgt_table.lookup(feature["target"])

  # Convert to int32
  feature["source"] = tf.to_int32(feature["source"])
  feature["target"] = tf.to_int32(feature["target"])
  feature["source_length"] = tf.to_int32(feature["source_length"])
  feature["target_length"] = tf.to_int32(feature["target_length"])
  feature["source_length"] = tf.squeeze(feature["source_length"], 1)
  feature["target_length"] = tf.squeeze(feature["target_length"], 1)

  return feature


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
  
  nmt_config = modeling.NmtConfig.from_json_file(FLAGS.nmt_config_file, vocab_size=len(vocabulary))

  vocabulary[0] = nmt_config.eos.encode()
  vocabulary[1] = nmt_config.unk.encode()
  vocabulary[2] = nmt_config.bos.encode()

  tf.gfile.MakeDirs(FLAGS.output_dir)

  with tf.gfile.Open(os.path.join(FLAGS.output_dir, 'model_config.json'), "w") as fout:
    fout.write(nmt_config.to_json_string())
  
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      nmt_config=nmt_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size)
  
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  train_input_fn = input_fn_builder(
      source_input_file=FLAGS.source_input_file,
      target_input_file=FLAGS.target_input_file,
      vocabulary=vocabulary,
      max_seq_length=FLAGS.max_seq_length,
      eos=nmt_config.eos.encode(),
      unkId=nmt_config.unkId)
  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)


if __name__ == "__main__":
  flags.mark_flag_as_required("source_input_file")
  flags.mark_flag_as_required("target_input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("nmt_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()