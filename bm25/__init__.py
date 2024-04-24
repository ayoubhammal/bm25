import gc
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BM25Score(BaseEstimator, ClassifierMixin):
    def __init__(self, vectorized_docs, k1=1.5, b=0.75, batch_size=50):
        # vectorized_docs: word counts
        self.k1 = k1
        self.b = b
        self.batch_size = batch_size
        self.vectorized_docs = vectorized_docs

    def fit(self, vectorized_queries=None, query_ids=None, *args):
        # args are for grid search integration
        self.n_d = self.vectorized_docs.sum(axis=1).reshape(-1, 1).A
        self.avgdl = np.mean(self.n_d)
        self.n_docs = self.vectorized_docs.shape[0]

        self.nq = np.sum(self.vectorized_docs > 0, axis=0).reshape(1, -1)
        self.idf = np.log(((self.n_docs - self.nq + 0.5) / (self.nq + 0.5)) + 1).A
        return self

    def predict(self, vectorized_queries):
        # bigger batches => faster but more memory heavy
        n_queries = vectorized_queries.shape[0]
        
        final_scores = []
        for batch_idx in range(int(n_queries / self.batch_size) + 1):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, n_queries)
            if start_idx >= end_idx:
                break
            batch_vectorized_queries = vectorized_queries[start_idx: end_idx]
            idx_tokens = np.argwhere(batch_vectorized_queries.sum(axis=0) > 0).reshape(-1)
            batch_vectorized_queries = batch_vectorized_queries[:, idx_tokens]
            vectorized_docs = self.vectorized_docs[:, idx_tokens].toarray()
            idf = self.idf[:, idx_tokens]
            
            scores = idf * (
                (vectorized_docs * (self.k1 + 1)) / \
                (
                    vectorized_docs + \
                    self.k1 * (1 - self.b + self.b * (self.n_d / self.avgdl))
                )
            )
            final_scores.append((batch_vectorized_queries > 0).astype(np.int8) @ scores.T)
           
            del vectorized_docs
            del scores
            gc.collect()
        return np.concatenate(final_scores, axis=0)
