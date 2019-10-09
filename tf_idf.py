import math

def tf_idf(corpus, num_return = 5):

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

    weights = []

    for r in range(len(corpus)):
        to_append = []

        for c in range(len(bag_of_words_arr[r])):
            tf = bag_of_words_arr[r][c]
            idf = math.log((len(corpus)/num_docs_per_word[c])**20)
            to_append.append(tf * idf)

        weights.append(to_append)

    cum_weights = []
    for col in range(len(weights[0])):
        sum = 0
        for row in range(len(weights)):
            sum += weights[row][col]
        cum_weights.append(sum)

    for i in range(len(word_list)):
        print(word_list[i],cum_weights[i], num_docs_per_word[i])

    to_return = []
    for i in range(num_return):
        max = 0
        max_i = 0
        for index in range(len(word_list) - i):
            #print("Evaluating", word_list[i])
            if(cum_weights[index] > max):
                #print("new max found at", index, "word is",word_list[index])
                max = cum_weights[index]
                max_i = index

        #print("Adding", word_list[max_i], "to return list, weight:", cum_weights[max_i])
        to_return.append(word_list[max_i])
        word_list[max_i], word_list[-i - 1] = word_list[-i - 1], word_list[max_i]
        cum_weights[max_i], cum_weights[-i - 1] = cum_weights[-i - 1], cum_weights[max_i]

    return to_return

if __name__ == "__main__":

    corpus = ["I was so hungry last night, I ate all the food in the fridge",
                "The food in the restaurant tasted great, I left a huge tip",
                "The pork ribs were delicious, I'll be eating there every week.",
                "I need to use a Linear Regression in order to predict Boston Housing",
                "My model classifies the emotions with a 90% accuracy",
                "My cancer dectection models needs to be optimized around a high recall score",
                "Waffles and fried chicken are a tasty combo!"]

    #print(tf_idf(corpus))
