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

TASK="test"

tag=10k

for CHECKPOINT in 20000
do

for ALPHA in 1.6
do

for BEAM_SIZE in 5
do

MODEL=transformer_${tag}
SOURCE=~/data/iwslt.${TASK}.de.${tag}.bpe
NMT_DIR=gs://experimental_research/research/iwslt/${MODEL}/model.ckpt-${CHECKPOINT}
CONFIG=gs://experimental_research/research/iwslt/${MODEL}/model_config.json
VOCAB=~/data/iwslt.de-en.${tag}.vocab

BATCH_SIZE=128

DECODE_LENGTH=20

TARGET=~/data/output/${MODEL}.${TASK}.decode_c${CHECKPOINT}_a${ALPHA}_b${BEAM_SIZE}.txt

echo ${TARGET}

if [ ! -f ${TARGET} ]; then
python translate_tpu.py \
  --source_input_file=${SOURCE} \
  --target_output_file=${TARGET} \
  --vocab_file=${VOCAB} \
  --nmt_config_file=${CONFIG} \
  --decode_batch_size=${BATCH_SIZE} \
  --decode_alpha=${ALPHA} \
  --decode_length=${DECODE_LENGTH} \
  --init_checkpoint=${NMT_DIR} \
  --beam_size=${BEAM_SIZE} \
  --tpu_name=${TPU_NAME} \
  --num_tpu_cores=8
fi

done
done
done