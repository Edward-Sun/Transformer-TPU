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
Shuffle Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy

def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle Dataset")

    parser.add_argument("--input", type=str, nargs=2, required=True, help="Dataset")
    parser.add_argument("--suffix", type=str, default="shuf", help="Suffix of Output File")
    parser.add_argument("--seed", default=12345, type=int, help="Random Seed")

    return parser.parse_args()

def main(args):
    if args.seed:
        numpy.random.seed(args.seed)

    with open(args.input[0], "r") as frs:
        with open(args.input[1], "r", errors='ignore') as frt:
            data1 = [line for line in frs]
            data2 = [line for line in frt]

    if len(data1) != len(data2):
        raise ValueError("length of two files are not equal")

    indices = numpy.arange(len(data1))
    numpy.random.shuffle(indices)

    with open(args.input[0] + "." + args.suffix, "w") as fws:
        with open(args.input[1] + "." + args.suffix, "w") as fwt:
            for idx in indices.tolist():
                fws.write(data1[idx])
                fwt.write(data2[idx])

if __name__ == "__main__":
    main(parse_args())