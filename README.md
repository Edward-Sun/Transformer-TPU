# Transformer-TPU

Course Assignment 1 of 11731: MT & Seq2Seq Models

Author: Zhiqing Sun (zhiqings@andrew.cmu.edu)

Course Instructor: Graham Neubig

**Introduction**

This is a high-performance TensorFlow implementation of Transformer with TPU support. we refer the BERT implementation (https://github.com/google-research/bert) when writing *train.py*, and refer the tensor2tensor implementation (https://github.com/tensorflow/tensor2tensor) and the THUMT implementation (https://github.com/THUNLP-MT/THUMT) when writing *translate.py* and the codes in the *models* directory.

**Pre-Process**

We use bpe (https://github.com/rsennrich/subword-nmt) to preprocess the tokenized parallel corpora.

```bash
bash bpe.sh
bash shuffle_dataset.sh
```

**Train**

We use estimator with TPU to train a Transformer small on IWSLT14 De-En, which basically further calls train.py.

```bash
bash train.sh
```

The usage of train.py:

```bash
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
  --num_train_steps=1000000 \
  --save_checkpoints_steps=20000 \
  --keep_checkpoint_max=0 \
  --iterations_per_loop=1000 \
  --use_tpu=True \
  --tpu_name=${TPU_NAME} \
  --brain_session_gc_seconds=86400 \
  --num_tpu_cores=8
```

Check argparse configuration at train.py for more arguments and more details.

**Evaluate**

We also use TPU to do prediction, which further calls translate_tpu.py

```bash
bash translate_tpu.sh
```

The usage of translate_tpu.py:

```
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
```

**Reproducing the state-of-the-art results**

To reprocude the results, you can just follow the above instructions.

| IWSLT                       | Valid | Test  |
| --------------------------- | ----- | ----- |
| ours                        | 35.79 | 33.31 |
| Actor-Critic [1]            | -     | 28.53 |
| Neural PBMT [2]             | -     | 30.08 |
| Minimum Risk Training [3]   | -     | 32.84 |
| Self-attention baseline [4] | -     | 34.40 |
| DynamicConv [4]             | -     | 35.20 |



[1] Dzmitry Bahdanau, Philemon Brakel, Kelvin Xu, Anirudh Goyal, Ryan Lowe, Joelle Pineau, Aaron Courville, and Yoshua Bengio. An Actor-Critic Algorithm for Sequence Prediction. In Proceedings of ICLR, 2017.

[2] Po-Sen Huang, Chong Wang, Sitao Huang, Dengyong Zhou, and Li Deng. Towards neural phrase-based machine translation. In Proceedings of ICLR, 2018.

[3] Sergey Edunov, Myle Ott, Michael Auli, David Grangier, and Marc’Aurelio Ranzato. Classical Structured Prediction Losses for Sequence to Sequence Learning. In Proceedings of NAACL, 2018.

[4] Felix Wu, Angela Fan, Alexei Baevski, Yann N. Dauphin, and Michael Auli. Pay less attention with lightweight and dynamic convlutions. In Proceedings of ICLR, 2019.
