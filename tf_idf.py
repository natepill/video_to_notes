import math

def tf_idf(corpus):

    dict = {}

    for sentence in corpus:

        words = sentence.split(" ")
        for word in words:

            if word not in dict:
                dict[word] = 1

    word_list = list(dict.keys())

    bag_of_words_arr = []

    for sentence in corpus:

        to_add = list()
        for item in word_list:
            to_add.append(0)

        words = sentence.split(" ")

        for word in words:

            i = 0
            for item in word_list:
                if item == word:
                    to_add[i] += 1

                i += 1
        bag_of_words_arr.append(to_add)

    #return bag_of_words_arr
    num_docs_per_word = []
    for c in range(len(bag_of_words_arr[0])):
        counter = 0
        for r in range(len(corpus)):
            if(bag_of_words_arr[r][c] > 0):
                counter += 1
        num_docs_per_word.append(counter)

    to_return = []

    for r in range(len(corpus)):
        to_append = []

        for c in range(len(bag_of_words_arr[r])):
            tf = bag_of_words_arr[r][c]
            idf = math.log(len(corpus)/num_docs_per_word[c])
            to_append.append(tf * idf)

        to_return.append(to_append)

    return to_return
