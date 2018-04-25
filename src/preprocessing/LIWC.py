import os

# noinspection PyClassHasNoInit


class LIWC:

    """
    LIWC word categories count


    """

    @staticmethod
    def load_lexicon(lexicon):

        word_list = []
        stemmed = []
        non_stemmed = []

        temp_path = os.path.abspath("../AdvancedML/data/lexicon/") + '\\' + lexicon + '.txt'

        with open(temp_path) as f:
            for line in f:
                word_list.append(line.rstrip('\n'))

        for word in word_list:
            if '*' in word:
                word = word.replace('*', '').rstrip()
                stemmed.append(word)
            else:
                word = word.rstrip()
                non_stemmed.append(word)

        return stemmed, non_stemmed

    def category_count(self, lexicon, stemmed_tokens, non_stemmed_tokens):
        """

        :param lexicon: Expects String. The LIWC category to check
        :param stemmed_tokens: Expects Dictionary. A dictionary with the stemmed - tokens
        :param non_stemmed_tokens: Expects Dictionary. The non-stemmed tokens.
        :return: integer (how many words are in the specific category)

        """
        stemmed, non_stemmed = self.load_lexicon(lexicon=lexicon)
        stemmed_count = [stemmed_tokens[word] for word in stemmed if word in stemmed_tokens]
        non_stemmed_count = [non_stemmed_tokens[word] for word in non_stemmed if word in non_stemmed_tokens]

        return sum(non_stemmed_count + stemmed_count)

    @staticmethod
    def get_dictionary_words(categories):
        word_set = []
        for cat in categories:
            temp_stemmed, temp_non_stemmed = LIWC.load_lexicon(cat)
            for word in temp_non_stemmed:
                word_set.append(word)

        return set(word_set)

    @staticmethod
    def get_words_captured(tokens, dictionary_words):
        word_count = [1 for word in tokens if word in dictionary_words]
        try:
            return float(sum(word_count)) / float(len(dictionary_words))
        except ZeroDivisionError:
            return 0


