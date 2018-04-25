# -*- coding: utf-8 -*-

from time import time
import os
import numpy as np
import pandas as pd
import ftfy

# from src.load_data.SqliteConnection import SqliteConnection
from src.preprocessing.LIWC import LIWC
from src.preprocessing.MyBasicNLP import MyBasicNLP
from src.preprocessing.MyTextPosTags import MyTextPosTags
from src.preprocessing.TextStatistics import TextStatistics


path = '../AdvancedML/data/'

# cursor = SqliteConnection("UNBFakeNews").open_connection()
datasets = ['3.pkl', '4.pkl', '5.pkl', '6.pkl', '7.pkl', '8.pkl', '9.pkl', '10.pkl']
#

categories = ["Affect", "Article", "Cause", "Certain", "Cogproc", "Discrep", "Insight", "Motion", "Negate",
              "Negemo", "They", "SheHe", "Prep", "Ppron", "Posemo", "I", "We", "See", "Hear", "Social",
              "Space", "Tentat", "Time", "Conj", "Excl", "Incl", "FocusPresent", "FocusFuture",
              "FocusPast", "Differ", "Reward", "Risk", "Quant", "Compare", "Swear", "Netspeak", "Interrog", "You",
              "Conj", "Affiliation", "Power"]

pos_tags = ["NN", "NNP", "PRP", "PRP$", "WP", "DT", "WDT", "CD", "RB", "UH", "VB", "JJ", "VBD", "VBG", "VBN"
            "VBP", "VBZ", "MD"]

total_dictionary_words = LIWC.get_dictionary_words(categories=categories)   # creates a set of words from the dictionary
# categories we used

t0 = time()  # Define starting time
for dataset in datasets:
    data = pd.read_pickle(path + dataset)
    df = pd.DataFrame(data)
    output = []  # output list
    check_encoding_errors = 0

    for index, item in df.iterrows():
        features = []
        text_id = index
        text = item[0]
        label = item[1]
        print dataset, " ", text_id
        if isinstance(text, str):
            try:
                text = unicode(text, "utf-8")
            except UnicodeError as e:
                print e
                check_encoding_errors = check_encoding_errors + 1
                continue

        features.append(int(text_id))
        nlp = MyBasicNLP(text=text)
        text_tokens_dictionary = nlp.get_normalized_tokens_as_dict()
        text_tokens_stemmed_dictionary = nlp.get_stemmed_tokens_as_dict()
        tokens = nlp.get_normalized_tokens()

        # Loop for LIWC features
        for category in categories:
            LIWC_count = LIWC().category_count(lexicon=category, stemmed_tokens=text_tokens_stemmed_dictionary,
                                               non_stemmed_tokens=text_tokens_dictionary)
            features.append(LIWC_count)

        # Loop for pos tags features
        for tag in pos_tags:
            tag_counter = MyTextPosTags(text_tokens_dictionary).get_pos_tag(tag=tag)
            features.append(tag_counter)

        # Various Functions
        def redundancy():
            my_count = LIWC().category_count(lexicon="Function", stemmed_tokens=text_tokens_stemmed_dictionary,
                                             non_stemmed_tokens=text_tokens_dictionary)
            try:
                result = float(my_count) / float(nlp.get_filtered_words_count())
                return result
            except ZeroDivisionError:
                return 0

        def content_word_diversity():
            try:
                return float(len(text_tokens_dictionary)) / float(sum(text_tokens_dictionary.values()))
            except ZeroDivisionError:
                return 0

        # Text Statistics Features
        ts = TextStatistics(text)

        features.append(ts.get_clauses())  # Total Clauses
        try: # Avg Clauses (Total Clauses / Total Sentences)
            features.append(ts.get_clauses() / ts.get_sentences())
        except ZeroDivisionError as e:
            print "Avg Clauses zero division error"
            features.append(0)

        features.append(ts.get_average_word_length())    # Average Word Length
        try:
            features.append(ts.get_unique_words()/ts.get_words())   # Lexical Word Diversity
        except ZeroDivisionError as e:
            print "Lexical Word Diversity zero division error"
            features.append(0)
        features.append(content_word_diversity())   # Content word Diversity
        features.append(ts.noun_phrases_count())    # Noun phrase count
        features.append(ts.noun_phrase_avg_length())    # Noun phrase average length
        features.append(redundancy())   # Redundancy
        features.append(ts.get_sentences())     # Get number of sentences
        features.append(ts.get_words())     # Get number of words
        features.append(ts.get_long_words())    # Get Big words
        features.append(LIWC.get_words_captured(tokens=tokens, dictionary_words=total_dictionary_words))    # %words
        # captured , dictionary words
        features.append(ts.get_syllables())     # syllable count
        features.append(ts.get_average_syllables_per_word())    # Syllables per word
        features.append(ts.get_flesh_kincaid())     # flesh kincaid
        features.append(ts.get_fog_index())     # fog index
        features.append(ts.get_smog_index())    # smog index
        features.append(ts.get_average_words_per_sentence())    # average words per sentence
        features.append(ts.noun_phrase_avg_length())    # noun phrase avg length
        features.append(nlp.stop_words_percent())   # stopwords percent
        features.append(ts.get_capital_words())     # capital words number

        features.append(int(label))  # text annotation
        output.append(features)


    print "Encoding errors", check_encoding_errors
    np.savetxt(path + "/" + dataset+".csv", output, fmt='%s', delimiter=",")
    # print df_out

t1 = time()

print "Time needed: ", t1-t0

# path = os.path.abspath("../data/")
# np.savetxt(path + "/" + "burgoon_features.csv", output, delimiter=",")

