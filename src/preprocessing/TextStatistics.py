#!/usr/bin/env python
# coding: utf8

from textacy import extract
from textacy import preprocess
from textacy import constants
import textacy
import spacy
import string


class TextStatistics:
    """
    TextStats is based in Spacy and textacy
    in dependencies you should install en_core_web_sm for english language

    """

    my_nlp = spacy.load('en_core_web_sm')

    def __init__(self, text):
        self.text = preprocess.fix_bad_unicode(text=text, normalization='NFC')
        self.doc = self.my_nlp(self.text)
        self.ts = textacy.text_stats.TextStats(self.doc)
        self.marks = lambda l1: sum([1 for x in l1 if x == '!'])
        self.punctuation = lambda l1, l2: sum([1 for x in l1 if x in l2])
        self.quotes = lambda l1: sum([1 for x in l1 if x == '"'])
        self.tokenizer = self.my_nlp.Defaults.create_tokenizer()
        self.tokens = self.tokenizer(self.text)

    def noun_phrases(self):
        """

        :return:
        """
        pattern = constants.POS_REGEX_PATTERNS['en']['NP']
        return list(extract.pos_regex_matches(self.doc, pattern))

    def noun_phrases_count(self):
        """

        :return:
        """
        return len(self.noun_phrases())

    def get_noun_phrases_words(self):
        """

        :return:
        """
        counts = []
        [counts.append(len(noun_phrase)) for noun_phrase in self.noun_phrases()]
        return sum(counts)

    def noun_phrase_avg_length(self):
        """
        In noun phrase tokens is included punctuation

        :return: float
        """
        try:
            return float(self.get_noun_phrases_words()) / float(len(self.noun_phrases()))
        except ZeroDivisionError as e:
            print "Noun phrase error: ", e
            return 0

    def verb_phrases(self):
        """

        :return:
        """
        pattern = constants.POS_REGEX_PATTERNS['en']['VP']
        return list(extract.pos_regex_matches(self.doc, pattern))

    def verb_phrases_count(self):
        """

        :return:
        """
        return len(self.verb_phrases())

    def get_verb_phrases_words(self):

        counts = []
        [counts.append(len(verb_phrase)) for verb_phrase in self.verb_phrases()]
        return sum(counts)

    def verb_phrase_avg_length(self):

        try:
            return float(self.get_verb_phrases_words()) / float(len(self.verb_phrases()))
        except ZeroDivisionError as e:
            print "Noun phrase error: ", e
            return 0

    def get_clauses(self):
        """

        :return:
        """
        pattern = textacy.constants.POS_REGEX_PATTERNS['en']['NP']
        pattern = pattern + '<VERB>'
        return len(list(textacy.extract.pos_regex_matches(self.doc, pattern)))

    def get_flesh_kincaid(self):
        try:
            return self.ts.flesch_kincaid_grade_level
        except ZeroDivisionError as e:
            print e
            return 0

    def get_fog_index(self):
        try:
            return self.ts.gunning_fog_index
        except ZeroDivisionError as e:
            print "Fog Index error: ", e
            return 0

    def get_smog_index(self):
        try:
            return self.ts.smog_index
        except ZeroDivisionError as e:
            print "Smog Index: ", e
            return 0

    def get_long_words(self):
        return self.ts.basic_counts['n_long_words']

    def get_chars(self):
        return self.ts.basic_counts['n_chars']

    def get_monosyllable_words(self):
        return self.ts.basic_counts['n_monosyllable_words']

    def get_polysyllable_words(self):
        return self.ts.basic_counts['n_polysyllable_words']

    def get_sentences(self):
        return self.ts.basic_counts['n_sents']

    def get_syllables(self):
        return self.ts.basic_counts['n_syllables']

    def get_unique_words(self):
        return self.ts.basic_counts['n_unique_words']

    def get_words(self):
        return self.ts.basic_counts['n_words']

    def get_average_syllables_per_word(self):
        """

        :return:
        """
        sylls = self.get_syllables()
        words = self.get_words()
        try:
            return float(sylls) / float(words)
        except ZeroDivisionError:
            return 0

    def get_average_words_per_sentence(self):
        """

        :return:
        """
        words = self.get_words()
        sents = self.get_sentences()
        try:
            return float(words) / float(sents)
        except ZeroDivisionError:
            return 0

    def get_exclamation_marks(self):
        """
        Needs improvement. Maybe Spacy provides such function
        :return:
        """
        return self.marks(self.text)

    def get_punctuation(self):
        """
        Needs improvement. Maybe Spacy provides such function
        :return:
        """
        return self.punctuation(self.text, set(string.punctuation))

    def get_quotes(self):
        """
        Needs improvement. Maybe Spacy provides such function
        :return:
        """
        return self.quotes(self.text)

    def get_capital_words(self):
        return sum([1 for word in self.tokens if str(word).isupper()])

    def get_average_word_length(self):
        try:
            return float(self.get_chars())/float(self.get_words())
        except ZeroDivisionError:
            return 0

    def get_pausality(self):
        try:
            return float(self.get_punctuation())/float(self.get_sentences())
        except ZeroDivisionError as e:
            print "pausality caused error: ", e
            return 0

    def get_lexical_word_diversity(self):
        try:
            return float(self.get_unique_words())/float(self.get_words())
        except ZeroDivisionError as e:
            print "Lexical word diversity: ",e
            return 0
