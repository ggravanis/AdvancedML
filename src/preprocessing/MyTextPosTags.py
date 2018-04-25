import nltk
from collections import Counter


class MyTextPosTags:

    """
    based on NLTK default TagSet
    http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    https://catalog.ldc.upenn.edu/docs/LDC99T42/tagguid1.pdf

    1. Coordinating conjunctions


    """

    def __init__(self, my_tokens):
        self.my_tokens = my_tokens

    def pos_tags_counter(self):
        """

        :return: dictionary
        """
        word_tag = nltk.pos_tag(self.my_tokens)
        counts = Counter(tag for word, tag in word_tag)
        return counts

    def get_pos_tag(self, tag):
        return self.pos_tags_counter()[tag]

    def get_coordinating_conjunctions(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['CC']

    def get_cardinal_numbers(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['CD']

    def get_determiners(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['DT']

    def get_existential_there(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['EX']

    def get_foreign_words(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['FW']

    def get_subordinating_conjunctions(self):
        """

        :return:
        """
        return self.pos_tags_counter()['IN']

    def get_adjectives(self):
        """
        JJ: Hyphenated compounds that are
            used as modifiers are tagged as adjectives
        :return: int
        """
        return self.pos_tags_counter()['JJ']

    def get_adjectives_comparative(self):
        """
        JJR: Adjectives with the comparative
        ending -er and a comparative meaning
        :return: int
        """
        return self.pos_tags_counter()['JJR']

    def get_adjectives_superlative(self):
        """
        JJS: Adjectives with the superlative ending -est
        :return:
        """
        return self.pos_tags_counter()['JJS']

    def get_item_markers(self):
        """

        :return:
        """
        return self.pos_tags_counter()['LS']

    def get_modal(self):
        """
         This category includes all verbs that
         don't take an -s ending in the third person
         singular present. e.g. (can, could, dare,
         may, might, must, ought, shall, should, will,
         would)

        :return: int
        """
        return self.pos_tags_counter()['MD']

    def get_noun_singular(self):
        """


        :return: int
        """
        return self.pos_tags_counter()['NN']

    def get_noun_plural(self):
        """

        :return:
        """
        return self.pos_tags_counter()['NNS']

    def get_proper_noun_singular(self):
        """

        :return:
        """
        return self.pos_tags_counter()['NNP']

    def get_proper_noun_plural(self):
        """

        :return:
        """
        return self.pos_tags_counter()['NNPS']

    def get_predeterminer(self):
        """

        :return:
        """
        return self.pos_tags_counter()['PDT']

    def get_possessive_ending(self):
        """

        :return:
        """
        return self.pos_tags_counter()['POS']

    def get_personal_pronouns(self):
        """

        :return:
        """
        return self.pos_tags_counter()['PRP']

    def get_possessive_pronouns(self):
        """

        :return:
        """
        return self.pos_tags_counter()['PRP$']

    def get_adverbs(self):
        """
        RB: this category includes most words that end
        in -ly as well as degree words like quite, too
        and very
        :return: int
        """
        return self.pos_tags_counter()['RB']

    def get_adverbs_comparative(self):
        """
        RBR: Adverbs with the comparative ending -er
        :return: int
        """
        return self.pos_tags_counter()['RBR']

    def get_adverbs_superlative(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['RBS']

    def get_particles(self):
        """

        :return: int
        """
        return self.pos_tags_counter()['RP']

    def get_symbols(self):
        """
    
        :return: int
        """
        return self.pos_tags_counter()['SYM']

    def get_to(self):
        """

        :return:
        """
        return self.pos_tags_counter()['TO']

    def get_interjection(self):
        """

        :return:
        """
        return self.pos_tags_counter()['UH']

    def get_verb(self):
        """

        :return:
        """
        return self.pos_tags_counter()['VB']

    def get_verb_past_tense(self):
        """

        :return:
        """
        return self.pos_tags_counter()['VBD']

    def get_verb_gerund(self):
        """
            counts the occurrences of verbs
            in gerund or present participle
            forms
        :return: int
        """
        return self.pos_tags_counter()['VBG']

    def get_verb_past_participle(self):
        """
            counts the occurrences of verbs
            in past participle forms
        :return: int
        """
        return self.pos_tags_counter()['VBN']

    def get_verb_non_third_person(self):
        """
            counts the occurrences of verbs
            in non-3rd person singular present
        :return: int
        """
        return self.pos_tags_counter()['VBP']

    def get_verb_third_person(self):
        """
            counts the occurrences of verbs
            in 3rd person singular present
            forms
        :return: int
        """
        return self.pos_tags_counter()['VBZ']

    def get_wh_determiner(self):
        """
            counts the occurrences of wh-determiners
        :return: int
        """
        return self.pos_tags_counter()['WDT']

    def get_wh_pronoun(self):
        """
            counts the occurrences of wh-pronouns
        :return: int
        """
        return self.pos_tags_counter()['WP']

    def get_possessive_wh_pronoun(self):
        """
            counts the occurrences of  possessive
            wh-pronouns
        :return: int
        """
        return self.pos_tags_counter()['WP$']

    def get_wh_adverb(self):
        """
            counts the occurrences of wh-averbs
        :return: int
        """
        return self.pos_tags_counter()['WRB']

    def get_rate_of_adjectives_adverbs(self):
        try:
            return float(self.pos_tags_counter()['JJ']) / self.pos_tags_counter()['RB']
        except ZeroDivisionError as e:
            print "Raa returned ",e
            return 0

    def get_emotiveness(self):
        # (total # of adjectives + total # of adverbs) / (total # of noun + total # of verbs)
        nominator = self.pos_tags_counter()['JJ'] + self.pos_tags_counter()['RB']
        denominator = self.pos_tags_counter()['NN'] + self.pos_tags_counter()['NNS'] + self.pos_tags_counter()['VB']
        try:
            return float(nominator)/denominator
        except ZeroDivisionError as e:
            print "Emotiveness returned ", e
            return 0

    def get_modifiers(self):
        return self.get_adjectives() + self.get_adverbs()

