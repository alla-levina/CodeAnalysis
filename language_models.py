import re
import nltk
from nltk.model.ngram import BaseNgramModel
from nltk.model.counter import NgramModelVocabulary
from nltk.model import count_ngrams
from collections import Counter
from nltk.model.counter import NgramCounter
from nltk.corpus import brown
import javalang
import glob



class MLENgramModel(BaseNgramModel):
    """Class for providing MLE ngram model scores.
    Inherits initialization from BaseNgramModel.
    """

    def score(self, word, context):
        """Returns the MLE score for a word given a context.
        Args:
        - word is expcected to be a string
        - context is expected to be something reasonably convertible to a tuple
        """
        context = self.check_context(context)
        if  self.ngrams[context].freq(word) == 0.0:
            return 0.00001
        return self.ngrams[context].freq(word)



class LidstoneNgramModel(BaseNgramModel):
    """Provides Lidstone-smoothed scores.
    In addition to initialization arguments from BaseNgramModel also requires
    a number by which to increase the counts, gamma.
    """

    def __init__(self, gamma, *args):
        super(LidstoneNgramModel, self).__init__(*args)
        self.gamma = gamma
        # This gets added to the denominator to normalize the effect of gamma
        self.gamma_norm = len(self.ngram_counter.vocabulary) * gamma

    def score(self, word, context):
        context = self.check_context(context)
        context_freqdist = self.ngrams[context]
        word_count = context_freqdist[word]
        ctx_count = context_freqdist.N()
        return (word_count + self.gamma) / (ctx_count + self.gamma_norm)



class LaplaceNgramModel(LidstoneNgramModel):
    """Implements Laplace (add one) smoothing.
    Initialization identical to BaseNgramModel because gamma is always 1.
    """

    def __init__(self, *args):
        super(LaplaceNgramModel, self).__init__(1, *args)