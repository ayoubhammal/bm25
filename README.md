# BM25 cpu implementation with batch capabilities

Compatible with scikit-learn.

## Usage

```python
vectorizer = CountVectorizer()
docs_v = vectorizer.fit_transform(docs)
queries_v = vectorizer.transform(queries)
bm25 = BM25Score(docs_v, batch_size=32).fit()
score = bm25.predict(queries_v)
```
