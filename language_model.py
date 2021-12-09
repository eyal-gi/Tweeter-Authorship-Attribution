import re
import sys
import random
import math
import collections
from collections import defaultdict
from nltk import TweetTokenizer


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a language model
        from a given text.
        It supports language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
    """

    def __init__(self, n=3, chars=False):
        """Initializing a language model object.
        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather then word tokens.
                          Defaults to False
        """
        self.n = n
        self.model_dict = defaultdict(
            int)  # a dictionary of the form {ngram:count}, holding counts of all ngrams in the specified text.
        self.chars = chars
        self.vocabulary = None  # a set of the types in the text
        self.ngrams_dict = defaultdict(int)  # dictionary of ngrams dictionaries
        self.tt = TweetTokenizer()

    def build_model(self, text):
        """populates the instance variable model_dict.

            Args:
                text (str): the text to construct the model from.
        """

        if self.chars is True:
            tokens = list(text)
            self.model_dict = defaultdict(int, collections.Counter(
                text[i:i + self.n] for i in range(len(tokens) - self.n + 1)))
            for j in range(self.n):
                self.ngrams_dict[self.n - j] = defaultdict(int, collections.Counter(
                    text[i:i + self.n - j] for i in range(len(tokens) - (self.n - j))))

        else:
            # tokens = re.split(r'\s+', text)  # create a list of words out of the corpora
            tokens = self.tt.tokenize(text=text)  # create a list of words out of the corpora
            # Every tuple of n words is joined to a string, and the Counter func creates a dict with counts
            self.model_dict = defaultdict(int, collections.Counter(
                " ".join(tuple(tokens[i:i + self.n])) for i in range(len(tokens) - self.n + 1)))
            # a dictionary of every possible n-gram dictionary. This is used for evaluation and generation with context smaller then n
            for j in range(self.n):
                self.ngrams_dict[self.n - j] = defaultdict(int, collections.Counter(
                    " ".join(tuple(tokens[i:i + self.n - j])) for i in range(len(tokens) - (self.n - j))))

        self.vocabulary = set(tokens)

    def get_model_dictionary(self):
        """Returns the dictionary class object
        """
        return self.model_dict

    def get_model_window_size(self):
        """Returning the size of the context window (the n in "n-gram")
        """
        return self.n

    def P(self, candidate, context, given_n=None):
        """Returns the probability of a given candidate word to follow the given context.
            By default, the functions calculates according to the model's n.
            In case a lower-n is needed, it can be provided and the function will calculate accordingly.

            Args:
                candidate(str): the candidate word to follow the context
                context(str): the context to follow
                given_n(int): if needed, a different n for the ngram

            Return:
                Float. The probability of the word for the context.

        """
        sequence = context.copy()  # the sequence begin with the context
        sequence.append(candidate)  # append the candidate word with the context

        # calculate according to the normal ngram algorithm
        if given_n is None:
            if self.chars is not True:
                return self.model_dict[" ".join(sequence)] / self.ngrams_dict[self.n - 1][" ".join(context)]
            else:
                return self.model_dict["".join(sequence)] / self.ngrams_dict[self.n - 1]["".join(context)]

        # calculate based on a different, given n.
        if self.chars is not True:
            return self.ngrams_dict[given_n][" ".join(sequence)] / self.ngrams_dict[given_n - 1][" ".join(context)]
        else:
            return self.ngrams_dict[given_n]["".join(sequence)] / self.ngrams_dict[given_n - 1]["".join(context)]

    def p_first(self, word):
        """Returns the probability for a given word to be the first in a context.
        As padding has not been implemented in this model, a calculation based on a word to appear first in a sentence
        is made.

            Args:
                word(str): the word to calculate its probability

            Return:
                Float. probability to be first
         """

        # if n is equal to 1, the probability is the word count above all words count.
        if self.n == 1:
            return self.model_dict[word] / sum(self.model_dict.values())

        # the probability is the word count above all possible ngrams count.
        else:
            return self.ngrams_dict[1][word] / sum(self.model_dict.values())

    def candidates(self, context, n_gram=None):
        """Returns a set of all possible ngrams sequences

            Args:
                context (list): the context to create candidates from
                n_gram(int): if generating for n_gram < self.n, generates for the relevant dictionary

            Return:
                List. The candidates words.
        """
        candi = set()  # initialize the candidates set

        for w in self.vocabulary:  # add every word in vocabulary to the given context
            c = context.copy()
            c.append(w)

            if n_gram is None:  # regular ngram
                if self.chars is not True:
                    if " ".join(c) in self.model_dict: candi.add(w)  # verify this sequence is in the model dictionary
                else:
                    if "".join(c) in self.model_dict: candi.add(w)
            else:  # different n for ngram.
                if self.chars is not True:
                    if " ".join(c) in self.ngrams_dict[n_gram]: candi.add(w)
                else:
                    if "".join(c) in self.ngrams_dict[n_gram]: candi.add(w)

        # candi1 = set(context+" "+w for w in self.vocabulary if context+" "+w in self.model_dict)
        return candi

    def generate_unigram(self, context, n):
        """Returns a string of the specified length based on words probabilities from the model's dictionary

            Args:
                context(string): The given context. Only used for the total word count, as the generation is not
                                    context based.
                n(int): The string length.

            Return:
                String. The generated text.

        """
        str = context
        for i in range(n - len(context)):
            # Randomly choose a word for the words distribution.
            str.append(
                random.choices(population=list(self.model_dict.keys()), weights=self.model_dict.values(), k=1)[0])

        if self.chars is not True:
            return " ".join(str)
        else:
            return ''.join(str)

    def generate(self, context=None, n=20):
        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted. If the length of the specified context exceeds (or equal to)
        the specified n, the method should return the a prefix of length n of the specified context.

        If context length is lower than self.n, a stupid backoff smoothing is applied to determine the next word.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.

        """

        if context == None: context = ''  # no context was given.
        if len(context) == 0:
            if self.n == 1:  # if this is a unigram model, generate from the unigram function
                empty = ['']
                return self.generate_unigram(empty, n)
            # choose a starting context randomly based on the context's distribution
            context = random.choices(population=list(self.ngrams_dict[self.n - 1].keys()),
                                     weights=self.ngrams_dict[self.n - 1].values())[0]
        if self.chars is True:
            context_l = list(context)
        else:
            context_l = self.tt.tokenize(context)  # the context as a list of words

        str = context_l.copy()  # the generates string start for the context

        # if the context is longer than n, returns a subset string.
        if len(context_l) >= n:
            if self.chars is not True:
                return " ".join(context_l[0:n])
            else:
                return ''.join(context_l[0:n])

        # if this is a unigram model, generate from the unigram function
        if self.n == 1: return self.generate_unigram(context_l, n)

        # context is shorter than the needed length
        if len(context_l) < self.n - 1:
            ngram = context_l.copy()
            for current_n in range(len(context_l) + 1, min(n, self.n)):  # generate enough words for context for ngram
                cands = self.candidates(ngram, current_n)
                if not cands:
                    if self.chars is not True:
                        return " ".join(str)  # no possible candidates -> end function.
                    else:
                        return ''.join(str)
                chosen = self.choose(cands, ngram,
                                     current_n)  # choose the word with highest probability from the list of options for next word
                str.append(chosen)
                ngram.append(chosen)


        # context is longer than needed
        elif len(context_l) > self.n - 1:
            ngram = (context_l[len(context_l) - (self.n - 1):]).copy()

        # context is exactly self.n - 1
        else:
            ngram = context_l.copy()

        for i in range(0, n - len(context_l)):
            cands = self.candidates(ngram.copy())
            if not cands: break  # no possible candidates -> end function.
            chosen = self.choose(cands,
                                 ngram.copy())  # choose the word with highest probability from the list of options for next word
            str.append(chosen)
            ngram.append(chosen)
            ngram.pop(0)

        if self.chars is not True:
            return " ".join(str)
        else:
            return ''.join(str)

    def choose(self, candidates, context_, n_gram=None):
        """Return the word with the highest probability to be next in the sentence based on ngrams.
        If there are more than one word with the name probability, a random choice is made.

        Args:
            candidates (list): list of possible candidates, based on ngram algorithm

        Return:
            The chosen word (str)
        """
        context = context_.copy()

        probs = {}  # dictionary of the candidates with their probabilities
        for c in candidates:  # c is a tuple of (ngram list, predicted word)
            probs[c] = self.P(c, context, n_gram)  # calculate the probability for each candidate.

        return (random.choices(population=list(probs.keys()), weights=probs.values(), k=1))[0]

    def evaluate(self, text):
        """Returns the log-likelihood of the specified text to be a product of the model.
           Laplace smoothing should be applied if necessary.

           Words that are OOV (out of vocabulary) or context that doesn't appear in the model dictionary are treated by
           laplace smoothing. If laplace smoothing has been applied, all further words will be smoothed.
           The first n-1 words are evaluated by stupid backoff smoothing.

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        apply_smoothing = False  # states is smoothing has been applied

        if not text:  # no text to evaluate
            raise Exception('A test must be inserted for evaluation')
        text = normalize_text(text)
        log_probs = []

        if self.chars is not True:
            text_list = self.tt.tokenize(text)  # split the text to a list of words
        else:
            text_list = list(text)

        text_len = len(text_list)  # text word count

        # Unigram model
        if self.n == 1:
            for w in text_list:
                if w in self.vocabulary and apply_smoothing == False:
                    log_probs.append(math.log(self.model_dict[w] / sum(self.model_dict.values())))
                else:  # word is OOV
                    apply_smoothing = True
                    log_probs.append(math.log(self.smooth(w, 1)))
            return round(sum(log_probs), 3)

        # Evaluation for the first word:
        if text_list[0] in self.vocabulary:  # word from vocabulary
            log_probs.append(
                math.log(self.p_first(text_list[0])))  # calc first word based on the model's context distribution
        else:  # word OOV
            log_probs.append(math.log(self.smooth(text_list[0], 1)))
            apply_smoothing = True

        for n in range(2, min(text_len,
                              self.n)):  # calc probability for words until there are enough for ngram according to self.n
            if self.chars is not True:
                ngram = " ".join(text_list[0:n])
                nm_gram = " ".join(text_list[0:n - 1])
            else:
                ngram = "".join(text_list[0:n])
                nm_gram = "".join(text_list[0:n - 1])

            if ngram in self.ngrams_dict[n] and apply_smoothing == False:  # word from vocab and no smoothing
                log_probs.append(math.log(self.ngrams_dict[n][ngram] / self.ngrams_dict[n - 1][nm_gram]))
            else:  # word is OOV or smoothing was applied
                apply_smoothing = True
                log_probs.append(math.log(self.smooth(ngram, nm_gram, n)))

        if text_len >= self.n:
            for i in range(0, text_len - (self.n - 1)):
                if self.chars is not True:
                    ngram = " ".join(text_list[i:i + self.n])
                    nm_gram = " ".join(text_list[i:i + self.n - 1])
                else:
                    ngram = "".join(text_list[i:i + self.n])
                    nm_gram = "".join(text_list[i:i + self.n - 1])

                if ngram in self.model_dict and apply_smoothing == False:
                    log_probs.append(math.log(self.model_dict[ngram] / self.ngrams_dict[self.n - 1][nm_gram]))
                else:
                    apply_smoothing = True
                    log_probs.append(math.log(self.smooth(ngram, nm_gram)))

        return round(sum(log_probs), 3)

    def smooth(self, ngram, nm_gram=None, given_n=None):
        """Returns the smoothed (Laplace) probability of the specified ngram.

            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
        """
        # Initialize count values
        c_ngram = 0
        c_context = 0

        # default model
        if given_n is None:
            if ngram in self.model_dict:  # ngram from vocabulary
                c_ngram = self.model_dict[ngram]
            if nm_gram in self.ngrams_dict[self.n - 1]:  # n-1 gram from vocabulary
                c_context = self.ngrams_dict[self.n - 1][nm_gram]
        # not default model
        else:
            if ngram in self.ngrams_dict[given_n]:  # ngram from vocabulary
                c_ngram = self.ngrams_dict[given_n][ngram]
            if nm_gram in self.ngrams_dict[given_n - 1]:  # n-1 gram from vocabulary
                c_context = self.ngrams_dict[given_n - 1][nm_gram]

        if given_n is None:
            v = len(self.ngrams_dict[self.n - 1])  # default model
        else:
            v = len(self.ngrams_dict[given_n - 1])  # not default model

        # Unigram model
        if given_n is not None and given_n == 1:
            return (c_ngram + 1) / (len(self.vocabulary) + 1)

        return (c_ngram + 1) / (c_context + v)


def normalize_text(text):
    """Returns a normalized version of the specified string.
      You can add default parameters as you like (they should have default values!)
      You should explain your decisions in the header of the function.

      Args:
        text (str): the text to normalize

      Returns:
        string. the normalized text.
    """
    nt = text.lower()  # lower-case the text
    # nt = re.sub('(?<! )(?=[.,:?!@#$%^&*()\[\]\\\])|(?<=[.,:?!@#$%^&*()\[\]\\\])(?! )', r' ', nt)
    nt = re.sub('(?<! )(?=[.,:?!@#$%^&*()\[\]\\\])|(?<=[.,:?!@#$%^&*()\[\]\\\])(?! )', r' ', nt)
    # tokens = self.tt.tokenize(nt)  # create a list of words out of the corpora
    # if tokens[-1] == '':
    #     tokens.pop()
    #
    # nt = " ".join(tokens)
    return nt
