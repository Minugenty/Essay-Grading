
import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import collections
from collections import defaultdict
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.linear_model import LinearRegression


dataframe = pd.read_csv('C:/Users/ASUS/Documents/VSCode/AutomatedEssayScorer/essays_and_scores.csv', encoding='latin-1')

dataframe = dataframe.dropna(axis=0, how='any')
dataframe = dataframe.dropna(axis=1, how='any')

x = dataframe[['essay_set', 'essay', 'domain1_score']]

x1 = x[x['essay_set'] == 1]
x2 = x[x['essay_set'] == 2]
x3 = x[x['essay_set'] == 3]
x4 = x[x['essay_set'] == 4]
x5 = x[x['essay_set'] == 5]
x6 = x[x['essay_set'] == 6]
x7 = x[x['essay_set'] == 7]
x8 = x[x['essay_set'] == 8]


scaler = MinMaxScaler()

#x1[['domain1_score']] = scaler.fit_transform(x1[['domain1_score']])
#x2[['domain1_score']] = scaler.fit_transform(x2[['domain1_score']])
#x3[['domain1_score']] = scaler.fit_transform(x3[['domain1_score']])
#x4[['domain1_score']] = scaler.fit_transform(x4[['domain1_score']])
#x5[['domain1_score']] = scaler.fit_transform(x5[['domain1_score']])
#x6[['domain1_score']] = scaler.fit_transform(x6[['domain1_score']])
#x7[['domain1_score']] = scaler.fit_transform(x7[['domain1_score']])
#x8[['domain1_score']] = scaler.fit_transform(x8[['domain1_score']])

data = pd.concat([x1, x2, x3, x4, x5, x6, x7, x8])

# Tokenize a sentence into words


def sentence_to_wordlist(raw_sentence):

    clean_sentence = re.sub("[^a-zA-Z0-9]", " ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)

    return tokens

# tokenizing an essay into a list of word lists


def tokenize(essay):
    stripped_essay = essay.strip()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)

    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))

    return tokenized_sentences

# calculating average word length in an essay


def avg_word_len(essay):

    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)

    return sum(len(word) for word in words) / len(words)

# calculating number of words in an essay


def word_count(essay):

    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)

    return len(words)

# calculating number of characters in an essay


def char_count(essay):

    clean_essay = re.sub(r'\s', '', str(essay).lower())

    return len(clean_essay)

# calculating number of sentences in an essay


def sent_count(essay):

    sentences = nltk.sent_tokenize(essay)

    return len(sentences)

# calculating number of punctuations in an essay


def punctuation_count(essay):

    clean_essay = re.sub(r'[a-zA-Z0-9]', ' ', essay)
    punctuations = nltk.word_tokenize(clean_essay)

    return len(punctuations)


# calculating number of lemmas per essay
def count_lemmas(essay):

    tokenized_sentences = tokenize(essay)

    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:

            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(
                    token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(
                    token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(
                    token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(
                    token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(
                    token_tuple[0], pos))

    lemma_count = len(set(lemmas))

    return lemma_count

# checking number of misspelled words


def count_spell_error(essay):

    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    data = open('C:/Users/ASUS/Documents/VSCode/AutomatedEssayScorer/big.txt').read()

    words_ = re.findall('[a-z]+', data.lower())

    word_dict = collections.defaultdict(lambda: 0)

    for word in words_:
        word_dict[word] += 1

    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    mispell_count = 0

    words = clean_essay.split()

    for word in words:
        if not word in word_dict:
            mispell_count += 1

    return mispell_count


# parts of speech - calculating number of nouns, adjectives, verbs, adverbs, pronouns and prepositions in an essay


def count_pos(essay):

    tokenized_sentences = tokenize(essay)

    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    pronoun_count = 0
    preposition_count = 0

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
            elif pos_tag.startswith('P'):
                pronoun_count += 1
            elif pos_tag.startswith('I'):
                preposition_count += 1

    return noun_count, adj_count, verb_count, adv_count, pronoun_count, preposition_count

# extracting essay features


def extract_features(data):

    features = data.copy()

    features['char_count'] = features['essay'].apply(char_count)

    features['word_count'] = features['essay'].apply(word_count)

    features['sent_count'] = features['essay'].apply(sent_count)

    features['avg_word_len'] = features['essay'].apply(avg_word_len)

    features['lemma_count'] = features['essay'].apply(count_lemmas)

    features['spell_err_count'] = features['essay'].apply(count_spell_error)

    features['punctuation_count'] = features['essay'].apply(punctuation_count)

    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'], features[
        'pronoun_count'], features['preposition_count'] = zip(*features['essay'].map(count_pos))

    return features


# extracting features from essay set 1
# features_set = extract_features(data)
# features_set.to_csv("features_set.csv")


# read extracted features
features_set = pd.read_csv('C:/Users/ASUS/Documents/VSCode/AutomatedEssayScorer/features_set.csv', encoding='latin-1')


# splitting data

X = features_set.iloc[:, 4:].values

y = features_set['domain1_score'].values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVR()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)




# Save model
joblib.dump(clf,'model1.pkl')
