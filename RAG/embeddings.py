import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class LocalTextEmbedder:
    def __init__(self, n_features: int = 1024):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            analyzer="char_wb",
            ngram_range=(3, 5),
            norm=None,
        )

    def encode(self, texts, convert_to_numpy: bool = True):
        single_text = isinstance(texts, str)
        items = [texts] if single_text else list(texts)
        matrix = self.vectorizer.transform(items)
        vectors = matrix.toarray().astype(np.float32)
        if convert_to_numpy:
            return vectors[0] if single_text else vectors
        return vectors[0].tolist() if single_text else vectors.tolist()
