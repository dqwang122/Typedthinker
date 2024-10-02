import os
import torch
import math
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util



from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)

class Memory:
    def __init__(self, corpus, memory_size=1000, device='cuda'):

        self.memory_size = memory_size
        self.device = device

        self.memory = corpus[:memory_size]  # List of tuples (query, response), there might be duplicates queries
        self.query_embeddings = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.initialize(self.memory)

    def initialize(self, corpus):
        query = [m[0] for m in corpus]
        query_embeddings = self.embedder.encode(query, convert_to_tensor=True)
        query_embeddings = query_embeddings.to(self.device)
        query_embeddings = normalize_embeddings(query_embeddings)
        self.query_embeddings = query_embeddings
    
    def add(self, query, response):
        self.memory.append((query, response))
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]
        self.initialize(self.memory)
    
    def retrieve(self, query, top_k=1, thre=0, batch_size=32, verbose=False):
        hits = []
        for i in tqdm(range(0, len(query), batch_size), total=math.ceil(len(query) // batch_size),
                      desc=f"Retrieving responses with semantic search (Top: {top_k} & thre: {thre})"):
            batch_query = query[i: i + batch_size]
            embed_query = self.embedder.encode(batch_query, convert_to_tensor=True)
            embed_query = embed_query.to(self.device)
            embed_query = normalize_embeddings(embed_query)
            batch_hits = semantic_search(embed_query, self.query_embeddings, score_function=dot_score, top_k=top_k)
            hits.extend(batch_hits)

            if verbose:
                print(batch_hits)

        retrieved_responses = []
        for q, r in zip(query, hits):
            corpus_id = [x['corpus_id'] for x in r if x['score'] > thre]
            retrieved = [self.memory[i] for i in corpus_id]
            retrieved_responses.append(retrieved)
        return retrieved_responses


if __name__ == '__main__':
    # file = "dataset/store/logicqa.analogical.jsonl"
    # df = pd.read_json(file, orient='records', lines=True)
    # corpus = list(zip(df['question'].values,df['solution'].values))
    # mem = Memory(corpus, memory_size=20000, device='cuda')
    # query = [
    #     "Find a movie similar to The Usual Suspects, Interview with the Vampire The Vampire Chronicles, The Shawshank Redemption, Pulp Fiction:\nOptions:\n(A) Toy Soldiers\n(B) The Fugitive\n(C) The Wasp Woman\n(D) Baxter",
    #     "Today, William went to the art studio. Between what times could they have gone?\nWe know that:\nWilliam woke up at 8am.\nSteven saw William working at the office from 8am to 10am.\nDavid saw William working out at the gym from 10am to 3pm.\nKimberly saw William taking photos near the Leaning Tower of Pisa from 3pm to 5pm.\nMary saw William buying clothes at the mall from 6pm to 8pm.\nSean saw William taking photos near the Eiffel Tower from 8pm to 10pm.\nThe art studio was closed after 10pm.\nBetween what times could William have gone to the art studio?\nOptions:\n(A) 8pm to 10pm\n(B) 6pm to 8pm\n(C) 10am to 3pm\n(D) 5pm to 6pm"
    # ]
    
    # ret = mem.retrieve(query, top_k=3, thre=0)
    # print(ret)
    benchmark="logicqa"
    for split in ["valid", "test"]:
        file = f"dataset/0417/{benchmark}.{split}.jsonl"
        data = [json.loads(x) for x in open(file)]
        print(file, len(data))

        from retrieval import Memory
        corpus_file = f"dataset/store/{benchmark}.raw.jsonl"
        refer = pd.read_json(corpus_file, orient='records', lines=True)
        print(f"Load {refer.shape[0]} memory from {corpus_file}...")
        corpus = list(zip(refer['question'].values,refer['solution'].values))
        explicit_mem = Memory(corpus, memory_size=30000, device='cuda')
        queries = [d['question'] for d in data]
        retrieved = explicit_mem.retrieve(queries, top_k=1, batch_size=1000, verbose=False)

        save_file = f"dataset/{benchmark}.{split}.jsonl"
        fout = open(save_file, 'w')
        for d, r in zip(data, retrieved):
            r = r[0]
            d['retrieval'] = f"Question:{r[0]}\nAnswer:\\boxed{{{r[1]}}}."
            fout.write(json.dumps(d) + '\n')
        print(save_file)