"""Okapi BM25 over a sparse term matrix (the lexical baseline)."""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def s_stem(word):
    """Harman's S-stemmer (1991): only plural endings, so shoe and shoes match."""
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    return word


def tokenize_stemmed(text):
    return [s_stem(t) for t in tokenize(text)]


class BM25:
    def __init__(self, k1=1.2, b=0.75, stem=False):
        self.k1, self.b, self.stem = k1, b, stem

    def fit(self, docs):
        self.vec = CountVectorizer(tokenizer=tokenize_stemmed if self.stem else tokenize,
                                   lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(docs).tocsc().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        # doc x term matrix of BM25 term weights, idf folded in
        self.w_dt = sparse.csr_matrix((w * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        self.w = self.w_dt.T.tocsr()  # term x doc, for scoring the whole catalogue
        return self

    def score(self, query, doc_idx):
        """BM25 scores of the given documents for one query."""
        q = self.vec.transform([query])
        q.data[:] = 1.0
        return (self.w_dt[doc_idx] @ q.T).toarray().ravel()

    def search(self, queries, k=100, batch=256):
        """Returns (scores, indices), each of shape (n_queries, k)."""
        out_s, out_i = [], []
        for i in range(0, len(queries), batch):
            q = self.vec.transform(queries[i:i + batch])
            q.data[:] = 1.0  # a repeated query word does not count twice
            s = (q @ self.w).toarray()
            idx = np.argpartition(-s, k, axis=1)[:, :k]
            top = np.take_along_axis(s, idx, axis=1)
            order = np.argsort(-top, axis=1)
            out_i.append(np.take_along_axis(idx, order, axis=1))
            out_s.append(np.take_along_axis(top, order, axis=1))
        return np.vstack(out_s), np.vstack(out_i)
