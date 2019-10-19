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

TPU_NAME="tpuv3-demand-3"

tag=10k

SOURCE=gs://experimental_research/research/iwslt/iwslt.train.de.${tag}.bpe.shuf
TARGET=gs://experimental_research/research/iwslt/iwslt.train.en.${tag}.bpe.shuf
NMT_DIR=gs://experimental_research/research/iwslt/transformer_${tag}
VOCAB=~/data/iwslt.de-en.${tag}.vocab
CONFIG=~/11731/iwslt_config.json

BATCH_SIZE=32768
WARM_UP=4000
LEARNING_RATE=0.1

python train.py \
  --source_input_file=${SOURCE} \
  --target_input_file=${TARGET} \
  --vocab_file=${VOCAB} \
  --nmt_config_file=${CONFIG} \
  --max_seq_length=256 \
  --learning_rate=${LEARNING_RATE} \
  --output_dir=${NMT_DIR} \
  --train_batch_size=${BATCH_SIZE} \
  --num_warmup_steps=${WARM_UP} \
  --num_train_steps=115000 \
  --save_checkpoints_steps=20000 \
  --keep_checkpoint_max=0 \
  --iterations_per_loop=1000 \
  --use_tpu=True \
  --tpu_name=${TPU_NAME} \
  --brain_session_gc_seconds=86400 \
  --num_tpu_cores=8
