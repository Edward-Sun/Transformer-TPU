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

tag=10k

train_de_output=~/data/iwslt.train.de.${tag}.bpe
train_en_output=~/data/iwslt.train.en.${tag}.bpe

python3 shuffle_dataset.py --input ${train_de_output} ${train_en_output}

gsutil cp ${train_de_output}.shuf gs://experimental_research/research/iwslt/
gsutil cp ${train_en_output}.shuf gs://experimental_research/research/iwslt/