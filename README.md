# Fine Tune BERT for Text Classification with TensorFlow

This repository documents a binary text classification workflow that fine-tunes a pretrained BERT encoder in TensorFlow to distinguish **sincere** questions from **insincere** ones. The implementation uses TensorFlow 2, TensorFlow Hub, TensorFlow Model Garden BERT utilities, and a custom `tf.data` input pipeline built around the Quora Insincere Questions Classification dataset.

The notebook covers the full training path end to end: dataset ingestion, stratified sampling, BERT tokenization, TensorFlow input pipeline construction, model definition, supervised fine-tuning, and sample inference. The recorded run in the notebook uses a reduced stratified subset of the original dataset for faster experimentation, so the reported metrics should be interpreted as subset-level results rather than full-dataset benchmark numbers.

## Project Objectives

- Build TensorFlow input pipelines for text data with the `tf.data` API.
- Tokenize and preprocess raw text into BERT-compatible features.
- Fine-tune a pretrained BERT encoder for binary text classification.

## Dataset

The project uses the **Quora Insincere Questions Classification** dataset, a large-scale binary classification dataset where each example contains a question and a target label indicating whether the question is sincere (`0`) or insincere (`1`).

- Notebook-loaded dataset shape: `1,306,122` rows x `3` columns
- Observed columns from the notebook output:
  - `qid`
  - `question_text`
  - `target`
- Dataset source:
  - Original competition page: [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification/data)
  - Archived file used in the notebook: [train.csv.zip](https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip)

The target histogram shown in the notebook indicates a clear class imbalance, with sincere questions dominating the dataset. That imbalance is preserved during splitting by using stratified sampling.

## Implementation Pipeline

### 1. Environment and setup

The notebook records the following runtime details:

- TensorFlow version: `2.19.0`
- TensorFlow Hub version: `0.16.1`
- Eager execution: enabled
- GPU availability: detected
- Recorded accelerator: NVIDIA Tesla T4

The setup stage also performs several environment-specific dependency steps:

- installs `tf_keras`
- clones `tensorflow/models` at tag `v2.3.0`
- installs a modified version of the Model Garden requirements file with `tensorflow-addons` excluded
- appends the cloned `models` repository to `sys.path`
- adds a lightweight `tensorflow_addons` import fallback so the notebook can import Model Garden utilities without failing when the package is unavailable

### 2. Data loading and split strategy

The dataset is loaded directly into a pandas DataFrame from the archived compressed CSV file. For training and validation, the notebook intentionally works on a small stratified subset rather than the full dataset:

- training split: `9,795` rows
- validation split: `972` rows

This sampling is produced with `train_test_split(..., stratify=df.target.values)` to preserve label proportions across splits.

### 3. `tf.data` dataset construction

The notebook creates TensorFlow datasets on CPU with `tf.data.Dataset.from_tensor_slices`, using:

- `question_text` as the raw text input
- `target` as the binary label

This yields datasets of `(text, label)` pairs before BERT preprocessing is applied.

### 4. BERT tokenization and feature generation

The pretrained encoder is downloaded from TensorFlow Hub:

- model: [`bert_en_uncased_L-12_H-768_A-12/2`](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2)

The notebook resolves the model vocabulary and casing configuration directly from the Hub layer and constructs a `FullTokenizer` from TensorFlow Model Garden BERT utilities. Example tokenization from the notebook:

- tokens for `"hi, how are you doing?"`: `['hi', '##,', 'how', 'are', 'you', 'doing', '##?']`
- token ids: `[7632, 29623, 2129, 2024, 2017, 2725, 29632]`

Raw text is converted into BERT-ready features in two steps:

1. `classifier_data_lib.InputExample` wraps the raw text and label.
2. `classifier_data_lib.convert_single_example(...)` produces:
   - `input_ids`
   - `input_mask`
   - `segment_ids`
   - `label_id`

Because `Dataset.map` runs in graph mode, the Python-side preprocessing logic is wrapped with `tf.py_function`. The resulting tensors are assigned static shapes and returned in the feature dictionary expected by the model:

```python
{
    "input_word_ids": input_ids,
    "input_mask": input_mask,
    "input_type_ids": segment_ids
}
```

### 5. Input pipeline optimization

After mapping text into BERT features, the notebook builds the final input pipeline with:

- `map(..., num_parallel_calls=tf.data.experimental.AUTOTUNE)`
- `shuffle(1000)` for training data
- `batch(32, drop_remainder=True)`
- `prefetch(tf.data.experimental.AUTOTUNE)`

The resulting dataset contract is:

- feature tensors:
  - `input_word_ids`: shape `(batch_size, 128)`
  - `input_mask`: shape `(batch_size, 128)`
  - `input_type_ids`: shape `(batch_size, 128)`
- label tensor:
  - scalar class id before batching
  - batched shape `(batch_size,)`

The notebook output confirms the batched `element_spec` for both training and validation as:

- `input_word_ids`: `TensorSpec(shape=(32, 128), dtype=tf.int32)`
- `input_mask`: `TensorSpec(shape=(32, 128), dtype=tf.int32)`
- `input_type_ids`: `TensorSpec(shape=(32, 128), dtype=tf.int32)`
- labels: `TensorSpec(shape=(32,), dtype=tf.int32)`

## Model Architecture

The classification model fine-tunes the full BERT encoder rather than freezing it.

- maximum sequence length: `128`
- batch size: `32`
- encoder: `hub.KerasLayer(..., trainable=True)`
- inputs:
  - `input_word_ids`
  - `input_mask`
  - `input_type_ids`

The Hub layer returns both:

- `pooled_output` with shape `(None, 768)`
- `sequence_output` with shape `(None, 128, 768)`

For classification, the notebook uses the pooled representation only. The classification head is intentionally simple:

- `Dropout(0.4)`
- `Dense(1, activation="sigmoid")`

According to the recorded model summary:

- total parameters: `109,483,010`
- trainable parameters: `109,483,009`
- non-trainable parameters: `1`

The model therefore exposes a single sigmoid output for binary classification. In the notebook's sample inference block, predictions are interpreted with a threshold of `0.7`.

## Training Configuration

The model is compiled with:

- optimizer: `Adam(learning_rate=2e-5)`
- loss: `BinaryCrossentropy()`
- metric: `BinaryAccuracy()`

Training configuration from the notebook:

- epochs: `4`
- full encoder fine-tuning: enabled (`trainable=True`)
- validation performed at the end of each epoch on the sampled validation split

## Results

The following metrics come from the stored notebook output and reflect training on the reduced stratified subset (`9,795` train / `972` validation), not the full `1.3M+` row dataset.

| Epoch | Train Loss | Train Binary Accuracy | Validation Loss | Validation Binary Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.1583 | 0.9420 | 0.1216 | 0.9542 |
| 2 | 0.0970 | 0.9645 | 0.1212 | 0.9583 |
| 3 | 0.0482 | 0.9823 | 0.1838 | 0.9583 |
| 4 | 0.0207 | 0.9940 | 0.2144 | 0.9573 |

Two patterns stand out in the recorded run:

- training loss decreases steadily across all four epochs
- validation loss is lowest at epoch 2, then rises while training accuracy continues to improve

This suggests the model begins to overfit after the second epoch on the sampled subset, even though validation accuracy remains high.

### Sample inference

The notebook also runs inference on five custom examples and classifies them with a `0.7` decision threshold:

| Example | Predicted Label |
| --- | --- |
| `May I please have your email id?` | `Sincere` |
| `The world is cruel, but it is also very beautiful.` | `Sincere` |
| `What a surprising turn of events!` | `Sincere` |
| An explicitly insulting sentence targeting others | `Insincere` |
| `The weather is nice today.` | `Sincere` |

This quick sanity check is consistent with the intended task: neutral or ordinary language is classified as sincere, while the hostile example is classified as insincere.

## Documented Input and Prediction Contract

The fine-tuned model consumes a dictionary of BERT features:

```python
{
    "input_word_ids": tf.Tensor(shape=(batch_size, 128), dtype=tf.int32),
    "input_mask": tf.Tensor(shape=(batch_size, 128), dtype=tf.int32),
    "input_type_ids": tf.Tensor(shape=(batch_size, 128), dtype=tf.int32)
}
```

It returns a single sigmoid score per example:

- output shape: `(batch_size, 1)`
- semantic meaning: probability-like score for the positive class (`insincere`)
- sample notebook threshold: values `>= 0.7` are interpreted as `Insincere`

Text preprocessing is not implemented with a custom tokenizer; it depends on TensorFlow Model Garden BERT utilities and the vocabulary/configuration resolved from the TensorFlow Hub BERT layer.

## Repository Contents

- `Fine_Tune_BERT_for_Text_Classification.ipynb`: the main notebook containing the full data preparation, BERT preprocessing, model definition, fine-tuning, evaluation, and inference workflow.
