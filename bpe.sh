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

num_operations=10000

tag=10k

train_de=~/cs11731-assignment1-template/data/train.de-en.de
train_en=~/cs11731-assignment1-template/data/train.de-en.en
valid_de=~/cs11731-assignment1-template/data/valid.de-en.de
valid_en=~/cs11731-assignment1-template/data/valid.de-en.en
test_de=~/cs11731-assignment1-template/data/test.de-en.de
test_en=~/cs11731-assignment1-template/data/test.de-en.en
train_file=~/data/iwslt.train.de-en.mix
train_file_output=~/data/iwslt.train.de-en.mix.${tag}.bpe
codes_file=~/data/iwslt.de-en.${tag}.code

train_de_output=~/data/iwslt.train.de.${tag}.bpe
train_en_output=~/data/iwslt.train.en.${tag}.bpe
valid_de_output=~/data/iwslt.valid.de.${tag}.bpe
valid_en_output=~/data/iwslt.valid.en.${tag}.bpe
test_de_output=~/data/iwslt.test.de.${tag}.bpe
test_en_output=~/data/iwslt.test.en.${tag}.bpe

vocab_file=~/data/iwslt.de-en.${tag}.vocab

cat ${train_de} ${train_en} > ${train_file}

subword-nmt learn-bpe -s ${num_operations} -o ${codes_file} < ${train_file}

subword-nmt apply-bpe -c ${codes_file} < ${train_file} > ${train_file_output}

subword-nmt get-vocab < ${train_file_output} > ${vocab_file}

subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${train_de} > ${train_de_output}
subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${train_en} > ${train_en_output}

subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${valid_de} > ${valid_de_output}
subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${valid_en} > ${valid_en_output}

subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${test_de} > ${test_de_output}
subword-nmt apply-bpe --vocabulary ${vocab_file} --vocabulary-threshold 5 -c ${codes_file} < ${test_en} > ${test_en_output}