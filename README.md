# xQG4xDR

The official repository for the paper [Augmenting Passage Representations with Query Generation
for Enhanced Cross-Lingual Dense Retrieval](https://arxiv.org/pdf/2305.03950.pdf), Shengyao Zhuang, Linjun Shou and Guido Zuccon, SIGIR2023.

> warning: the full pipeline of this codebase takes around 2.5TB disk space due to the large corpus and number of generated queries.

## Installation
We rely on [transformers](https://github.com/huggingface/transformers) and [Tevatron](https://github.com/texttron/tevatron/) DR training and inference toolkit.

First install Tevatron:
```bash
git clone https://github.com/texttron/tevatron/
cd tevatron
pip install --editable .
cd ..
```
> Note: this repo is tested with tevatron main branch with commit id: `6cd9f00277bc830bd31cad3a435bdd5ffb565c8c`.

Then install the dependencies for this project:
```bash
pip install -r requirements.txt
```

## xDR
### Training
We use the tevatron to train a cross-lingual dense retrieval model. The following command trains a model on the XOR-TyDi dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
  --output_dir model/bert-base-multilingual-cased-xor-tydi \
  --model_name_or_path bert-base-multilingual-cased \
  --save_steps 1000 \
  --dataset_name Tevatron/xor-tydi:eng_span/train \
  --fp16 \
  --per_device_train_batch_size 128 \
  --train_n_passages 8 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 128 \
  --num_train_epochs 40 \
  --logging_steps 100 \
  --passage_field_separator " " \
  --cache_dir cache \
  --report_to wandb \
  --run_name bert-base-multilingual-cased \
  --overwrite_output_dir
```

We take the last saved checkpoint as our xDR model.

If you do not want to train the model yourself, we provide our trained model on huggingface model hub: [ielabgroup/bert-base-multilingual-cased-xor-tydi](https://huggingface.co/ielabgroup/bert-base-multilingual-cased-xor-tydi). We use our trained model for the rest of this instruction.

### Evaluation
We use the tevatron to evaluate the xDR model on the XOR-TyDi dataset. The following commands are the full evaluation pipeline which includes encoding the corpus and queries, retrieving passages, and evaluating the retrieval results:
```bash
# encoding the corpus
mkdir encodings
for s in $(seq 0 17)
do
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ielabgroup/bert-base-multilingual-cased-xor-tydi \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --p_max_len 128 \
  --dataset_name ielab/xor-tydi-xqg-augmented \
  --encoded_save_path encodings/corpus_emb.${s}.pkl \
  --encode_num_shard 18 \
  --encode_shard_index ${s} \
  --cache_dir cache &
done

# encoding the queries
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path ielabgroup/bert-base-multilingual-cased-xor-tydi \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --dataset_name Tevatron/xor-tydi:eng_span/dev \
  --encoded_save_path encodings/query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry \
  --cache_dir cache

# retrieving passages
mkdir runs
python -m tevatron.faiss_retriever \
--query_reps encodings/query_emb.pkl \
--passage_reps 'encodings/corpus_emb.*.pkl' \
--depth 500 \
--batch_size -1 \
--save_text \
--save_ranking_to runs/xortydi_dev_mbert.tsv

# evaluating the retrieval results
# download dev queries
wc -l https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_retrieve_eng_span_v1_1.jsonl

python3 make_xortydi_submission.py \
--rank_file /scratch/project/neural_ir/arvin/DSI-QG/runs/xortydi_dev_mbert.tsv \
--query_file xor_dev_retrieve_eng_span_v1_1.jsonl \
--corpus_file ielab/xor-tydi-xqg-augmented \
--output_file runs/xortydi_dev_mbert.json

python3 eval_xor_retrieve.py \
--data_file xor_dev_retrieve_eng_span_v1_1.jsonl \
--pred_file runs/xortydi_dev_mbert.json
```


## xQG
### Training
We follow the open-source xQG training pipeline in [DSI-QG](https://github.com/ArvinZhuang/DSI-QG) to train our xQG model ([step 1](https://github.com/ArvinZhuang/DSI-QG#step-1) of the pipeline with [`google/mt5-base`](https://huggingface.co/google/mt5-base) model).

We also provide our trained xQG model on huggingface model hub: [ielabgroup/xor-tydi-docTquery-mt5-base](https://huggingface.co/ielabgroup/xor-tydi-docTquery-mt5-base). We use our trained model for the rest of this instruction.

<details>
<summary>Example of Inferencing xQG</summary>

```python
from transformers import pipeline

lang2mT5 = dict(
    ar='Arabic',
    bn='Bengali',
    fi='Finnish',
    ja='Japanese',
    ko='Korean',
    ru='Russian',
    te='Telugu'
)
PROMPT = 'Generate a {lang} question for this passage: {title} {passage}'

title = 'Transformer (machine learning model)'
passage = 'A transformer is a deep learning model that adopts the mechanism of self-attention, differentially ' \
          'weighting the significance of each part of the input (which includes the recursive output) data.'


model_name_or_path = 'ielabgroup/xor-tydi-docTquery-mt5-base'
input_text = PROMPT.format_map({'lang': lang2mT5['ja'],
                                'title': title,
                                'passage': passage})

generator = pipeline(model=model_name_or_path,
                     task='text2text-generation',
                     device="cuda:0",
                     )

results = generator(input_text,
                    do_sample=True,
                    max_length=64,
                    num_return_sequences=10,
                    )

for i, result in enumerate(results):
    print(f'{i + 1}. {result["generated_text"]}')
```
</details>

In addition, we also provided all our generated cross-lingual queries on huggingface dataset hub: [ielab/xor-tydi-xqg-augmented](https://huggingface.co/datasets/ielab/xor-tydi-xqg-augmented).

Specifically, in our dataset we have generated 5 queries for each language per passage in the corpus, result in 5 * 7 languages * 18m passages = 630m generated queries in total. That's a lot of computation!

## Augmenting XOR-TyDi with cross-lingual queries
```bash
# encoding all the generated queries
for i in $(seq 0 17); do
  mkdir -p encodings/d2q_corpus/d2q_corpus_${i}
  for q in $(seq 0 4); do
  for lang in ar bn fi ja ko ru te; do
    CUDA_VISIBLE_DEVICES=0 python encode.py \
      --output_dir=temp \
      --model_name_or_path ielabgroup/bert-base-multilingual-cased-xor-tydi \
      --dataset_name ielab/xor-tydi-xqg-augmented \
      --fp16 \
      --per_device_eval_batch_size 128 \
      --lang ${lang} \
      --q_ind ${q} \
      --encoded_save_path encodings/d2q_corpus/d2q_corpus_${i}/${l}_id${q}.pkl \
      --encode_num_shard 18 \
      --encode_shard_index ${i} \
      --q_max_len 32 \
      --cache_dir cache &
  done
  wait
  done
done

# augmenting passage embeddings with generated query embeddings
alpha=0.01
for i in $(seq 0 17); do
mkdir -p encodings/augmented_corpus_${alpha}
python fuse_embeds.py \
--passage_reps encodings/corpus_emb.${i}.pkl \
--query_files encodings/d2q_corpus/d2q_corpus_${i}/'*.pkl' \
--save_path encodings/augmented_corpus_${alpha}/augmented_corpus_emb.${i}.pkl \
--alpha ${alpha}
done

# retrieving with augmented corpus
python -m tevatron.faiss_retriever \
--query_reps encodings/query_emb.pkl \
--passage_reps encodings/augmented_corpus_${alpha}/'augmented_corpus_emb.*.pkl' \
--depth 500 \
--batch_size -1 \
--save_text \
--save_ranking_to runs/xortydi_dev_mbert_augmented_alpha_${alpha}.tsv

# evaluating the retrieval results
python3 make_xortydi_submission.py \
--rank_file runs/xortydi_dev_mbert_augmented_alpha_${alpha}.tsv \
--query_file xor_dev_retrieve_eng_span_v1_1.jsonl \
--corpus_file ielab/xor-tydi-xqg-augmented \
--output_file runs/xortydi_dev_mbert_augmented_alpha_${alpha}.json

python3 eval_xor_retrieve.py \
--data_file xor_dev_retrieve_eng_span_v1_1.jsonl \
--pred_file runs/xortydi_dev_mbert_augmented_alpha_${alpha}.json
```

We also provided our run files in `runs/` folder.

The results reproduced by this repo:

R@2kt:

| Model       | Ar    | Bn    | Fi    | Ja    | Ko    | Ru    | Te    | Avg   |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|
| mBERT       | 41.10 | 49.01 | 52.23 | 37.34 | 48.07 | 33.33 | 48.32 | 44.20 | 
| mBERT + xQG | 42.39 | 54.93 | 54.14 | 33.61 | 52.28 | 33.76 | 52.52 | 46.23 | 

R@5kt:

| Model       | Ar    | Bn    | Fi    | Ja    | Ko    | Ru    | Te    | Avg   |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|
| mBERT       | 49.19 | 57.57 | 58.60 | 42.74 | 57.54 | 41.35 | 55.88 | 51.84 | 
| mBERT + xQG | 51.46 | 60.53 | 58.28 | 43.57 | 58.60 | 40.93 | 60.08 | 53.35 | 