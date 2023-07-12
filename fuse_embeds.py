import pickle
import os
import numpy as np
import glob
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def pickle_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def sort_by_did(index):
    dids = []
    embeds = []
    for did, embed in zip(index[1], index[0]):
        dids.append(did)
        embeds.append(embed)
    dids, embeds = zip(*sorted(zip(dids, embeds)))
    return dids, np.vstack(embeds)


def main():
    parser = ArgumentParser()
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--query_files', required=True)

    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--save_path', required=True)

    args = parser.parse_args()
    doc_embeds, dids = pickle_load(args.passage_reps)
    total_hyqe_embeds = np.zeros((len(dids), 768), dtype=np.float32)
    query_files = glob.glob(args.query_files)
    logging.info(f"Found {len(query_files)} query files")
    logger.info(f"Augmenting for {args.passage_reps}")
    for f in tqdm(query_files):
        hyqe_embeds, hyqe_dids = pickle_load(f)
        assert dids == hyqe_dids, "docids not match"
        total_hyqe_embeds += hyqe_embeds
        del hyqe_dids, hyqe_embeds

    embeds = ((1-args.alpha) * doc_embeds) + (args.alpha * total_hyqe_embeds)

    pickle_save((embeds, dids), args.save_path)


if __name__ == '__main__':
    main()

