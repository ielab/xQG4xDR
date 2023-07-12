import json
import jsonlines
import argparse
from datasets import load_dataset
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


def read_jsonlines(eval_file_name):
    import jsonlines
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def handle_args():
    parser = argparse.ArgumentParser(description='Create XORQA submission file'
)
    parser.add_argument('--rank_file', required=True)
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--corpus_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--cache_dir', default='cache')
    args=parser.parse_args()
    return args


def load_corpus(corpus_file, cache_dir):
    data = load_dataset(corpus_file, cache_dir=cache_dir)['train']
    corpus_map = {}
    for item in tqdm(data):
        corpus_map[item['docid']] = item["text"]
    return corpus_map


def load_ranking(rank_file):
    rank_map = {}
    with open(rank_file, 'r') as f:
        for line in f:
            qid, docid, rank = line.strip().split('\t')
            if qid not in rank_map.keys():
                rank_map[qid] = []
            rank_map[qid].append(docid)
    return rank_map


def main():
    args = handle_args()
    input_data = read_jsonlines(args.query_file)
    qid2query = {item["id"]: (item["question"], item["lang"]) for item in input_data}
    rank_map = load_ranking(args.rank_file)
    corpus_map = load_corpus(args.corpus_file, args.cache_dir)

    results = []
    for qid in tqdm(rank_map.keys()):
        _, lang = qid2query[qid]
        ranking = rank_map[qid]
        ctxs = []
        for docid in ranking:
            ctxs.append(corpus_map[docid])
        results.append({"id": qid, "lang": lang, "ctxs": ctxs})

    with open(args.output_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
    print("Success...")