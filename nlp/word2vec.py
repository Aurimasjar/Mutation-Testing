# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize

import gensim
from gensim.models import Word2Vec


def testing_word2vec():
    #  Reads ‘parsedJavaTest.txt’ file
    sample = open("java/parsed/parsedJavaTest.txt")
    s = sample.read()

    # Replaces escape character with space
    f = s.replace("\n", " ")

    data = []

    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)

    # Create Skip Gram model
    model = gensim.models.Word2Vec(data, min_count=1, vector_size=128,
                                   window=5, sg=1)

    # Print example results
    print("Cosine similarity between 'keyword' " +
          "and 'identifier' - Skip Gram : ",
          model.wv.similarity('keyword', 'identifier'))

    print("Cosine similarity between 'keyword' " +
          "and 'separator' - Skip Gram : ",
          model.wv.similarity('keyword', 'separator'))

    print('model.wv.vectors', model.wv.vectors)
    print('model.wv.index_to_key', model.wv.index_to_key)
    parsedFilename = "nlp/exampleVectors/vectorsJavaTest.txt"

    fw = open(parsedFilename, 'w')
    for vector in model.wv.vectors:
        fw.write(str(vector) + '\n')
    fw.close()
    print('word2vec done')


def word2vec(parsed_mutant_code):
    data = []
    for element in sent_tokenize(parsed_mutant_code):
        temp = []
        # tokenize the element into parameters
        for param in word_tokenize(element):
            temp.append(param.lower())
        data.append(temp)

    # Create Skip Gram model
    model = gensim.models.Word2Vec(data, min_count=1, vector_size=128,
                                   window=5, sg=1)

    # fixme analyze how to normalize data and get vectors of equal size
    vectors = model.wv.vectors
    vectors = vectors[0:128]
    vectors = [[abs(v1) for v1 in v] for v in vectors]
    # print('vectors', vectors)

    return vectors


def get_vectors(mutants):
    for mutant in mutants:
        parsed_mutant_code = " ".join([str(item) for item in mutant.parsed_code])
        mutant.vector = word2vec(parsed_mutant_code)
