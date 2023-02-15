import re
import copy
from pymystem3 import Mystem
from itertools import chain
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc


class TextsTokenizer:
    """Tokenizer"""

    def __init__(self):
        self.stopwords = []
        self.patterns = re.compile("")
        self.m = Mystem()

    def texts2tokens(self, texts: [str]) -> [str]:
        """Lemmatization for texts in list. It returns list with lemmatized texts"""
        try:
            text_ = "\n".join(texts)
            text_ = re.sub(r"[^\w\n\s]", " ", text_)
            lm_texts = "".join(self.m.lemmatize(text_.lower()))
            return [lm_tx.split() for lm_tx in lm_texts.split("\n")][:-1]
        except TypeError as e:
            return []

    def add_stopwords(self, stopwords: [str]):
        """adding stop words into class"""
        self.stopwords = [" ".join(x) for x in self.texts2tokens(stopwords)]
        self.patterns = re.compile("|".join([r"\b" + tx + r"\b" for tx in self.stopwords]))

    def del_stopwords(self, stopwords: [str]):
        """adding stop words into class"""
        stopwords_del = [x for x in chain(*self.texts2tokens(stopwords))]
        self.stopwords = [w for w in self.stopwords if w not in stopwords_del]
        self.patterns = re.compile("|".join([r"\b" + tx + r"\b" for tx in self.stopwords]))

    def tokenization(self, texts: [str]) -> [[]]:
        """list of texts lemmatization with stop words deleting"""
        lemm_texts = self.texts2tokens(texts)
        if self.stopwords:
            return [self.patterns.sub(" ", " ".join(l_tx)).split() for l_tx in lemm_texts]
        else:
            return lemm_texts

    def __call__(self, texts: [str]):
        return self.tokenization(texts)


class TokensVectorsBoW:
    """"""

    def __init__(self, max_dict_size: int):
        self.dictionary = None
        self.max_dict_size = max_dict_size

    def tokens2corpus(self, tokens: []):
        """queries2vectors new_queries tuple: (text, query_id)
        return new vectors with query ids for sending in searcher"""

        if self.dictionary is None:
            gensim_dict_ = Dictionary(tokens)
            print("len gensim_dict:", len(gensim_dict_))
            assert len(gensim_dict_) <= self.max_dict_size, "len(gensim_dict) must be less then max_dict_size"
            self.dictionary = Dictionary(tokens)
        else:
            gensim_dict_ = copy.deepcopy(self.dictionary)
            gensim_dict_.add_documents(tokens)
            if len(gensim_dict_) <= self.max_dict_size:
                self.dictionary = gensim_dict_
        return [self.dictionary.doc2bow(lm_q) for lm_q in tokens]

    def tokens2vectors(self, tokens: []):
        """"""
        corpus = self.tokens2corpus(tokens)
        return [corpus2csc([x], num_terms=self.max_dict_size) for x in corpus]

    def __call__(self, new_tokens):
        return self.tokens2vectors(new_tokens)


class TokensVectorsTfIdf(TokensVectorsBoW):
    """"""

    def __init__(self, max_dict_size):
        super().__init__(max_dict_size)
        self.tfidf_model = None

    def model_fill(self, tokens: []):
        """"""
        assert self.tfidf_model is None, "the model is already filled"
        corpus = super().tokens2corpus(tokens)
        self.tfidf_model = TfidfModel(corpus)

    def tokens2vectors(self, tokens: []):
        """"""
        vectors = super().tokens2corpus(tokens)
        test_r = []
        for x in vectors:
            # print(x)
            test_r.append(corpus2csc([x], num_terms=self.max_dict_size))
        return test_r
        #return [corpus2csc([x], num_terms=self.max_dict_size) for x in self.tfidf_model[vectors]]

    def __call__(self, new_tokens):
        return self.tokens2vectors(new_tokens)

