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

REFERENCE=~/cs11731-assignment1-template/data/${TASK}.de-en.en

BATCH_SIZE=128

MODEL=transformer_${tag}

DECODE_LENGTH=20

TARGET=~/data/output/${MODEL}.${TASK}.decode_c${CHECKPOINT}_a${ALPHA}_b${BEAM_SIZE}.txt

echo ${TASK}_c${CHECKPOINT}_a${ALPHA}_b${BEAM_SIZE}_l${DECODE_LENGTH}

perl multi-bleu.perl ${REFERENCE} < ${TARGET}

done
done
done