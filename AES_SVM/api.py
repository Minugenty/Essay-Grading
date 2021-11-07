from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import sys
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
import re, math
import collections
from collections import defaultdict, Counter
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer 
import json
import language_check
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
import csv
import random

app = Flask(__name__)
random_prompt=""
def sentence_to_wordlist(raw_sentence):
    clean_sentence = re.sub("[^a-zA-Z0-9]", " ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)
    return tokens

def tokenize(essay):
    stripped_essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
    return tokenized_sentences

def avg_word_len(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    avg_word_len.avrg = sum(len(word) for word in words) / len(words)
    return sum(len(word) for word in words) / len(words)

def word_count(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    word_count.word = len(words)
    return len(words)

def char_count(essay):
    clean_essay = re.sub(r'\s', '', str(essay).lower())
    char_count.chars = len(clean_essay)
    return len(clean_essay)

def sent_count(essay):
    sentences = nltk.sent_tokenize(essay)
    sent_count.sent = len(sentences)
    return len(sentences)

def punctuation_count(essay):
    clean_essay = re.sub(r'[a-zA-Z0-9]', ' ', essay)
    punctuations = nltk.word_tokenize(clean_essay)
    punctuation_count.p = len(punctuations)
    return len(punctuations)

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
    count_lemmas.lemma_count = len(set(lemmas))
    return count_lemmas.lemma_count

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
    count_spell_error.mispell_count = 0
    words = clean_essay.split()
    for word in words:
        if not word in word_dict:
            count_spell_error.mispell_count += 1
    return count_spell_error.mispell_count

def count_pos(essay):
    tokenized_sentences = tokenize(essay)
    count_pos.noun_count = 0
    count_pos.adj_count = 0
    count_pos.verb_count = 0
    count_pos.adv_count = 0
    count_pos.pronoun_count = 0
    count_pos.preposition_count = 0
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
            if pos_tag.startswith('N'):
                count_pos.noun_count += 1
            elif pos_tag.startswith('J'):
                count_pos.adj_count += 1
            elif pos_tag.startswith('V'):
                count_pos.verb_count += 1
            elif pos_tag.startswith('R'):
                count_pos.adv_count += 1
            elif pos_tag.startswith('P'):
                count_pos.pronoun_count += 1
            elif pos_tag.startswith('I'):
                count_pos.preposition_count += 1
    return count_pos.noun_count, count_pos.adj_count, count_pos.verb_count, count_pos.adv_count, count_pos.pronoun_count, count_pos.preposition_count

def extract_features(data):
    features = data.copy()
    features['char_count'] = features['essay'].apply(char_count)
    features['word_count'] = features['essay'].apply(word_count)
    features['sent_count'] = features['essay'].apply(sent_count)
    features['avg_word_len'] = features['essay'].apply(avg_word_len)
    features['lemma_count'] = features['essay'].apply(count_lemmas)
    features['spell_err_count'] = features['essay'].apply(count_spell_error)
    features['punctuation_count'] = features['essay'].apply(punctuation_count)
    features['count_pos.noun_count'], features['count_pos.adj_count'], features['count_pos.verb_count'], features['count_pos.adv_count'], features[
        'count_pos.pronoun_count'], features['count_pos.preposition_count'] = zip(*features['essay'].map(count_pos))
    return features

def preprocess_data(doc_set):
    #tokens = tokenize(str(doc_set))
    #return tokens
    stemming = PorterStemmer()
    stop_words = set(stopwords.words('english')) 
    word_tokens = nltk.word_tokenize(str(doc_set).lower()) 
    word_tokens = word_tokens[6:]
    stemmed_list = [stemming.stem(word) for word in word_tokens]
    filtered_sentence = [] 
    for w in stemmed_list: 
        if w not in stop_words and len(w)>2: 
            filtered_sentence.append(w)
    return filtered_sentence

def prepare_corpus(doc_clean):
    #doc_clean = str(doc_clean)
    dictionary = corpora.Dictionary([doc_clean])
    return dictionary

def fn_tdm(docs):
    vec = CountVectorizer()
    docs=[docs]
    X = vec.fit_transform(docs)
    print(X)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    return df

def create_gensim_lsa_model(essay_new,number_of_topics,words):
    clean_text = preprocess_data(essay_new)
    print(clean_text)
    dictionary = prepare_corpus(clean_text)
    corpus = [dictionary.doc2bow(text) for text in [clean_text]]
    print(corpus)
    doc_term_matrix = fn_tdm(str(clean_text))
    print("doc_term_matrix")
    print(doc_term_matrix)
    lsamodel = LsiModel(corpus=corpus, num_topics=number_of_topics, id2word = dictionary)
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.get_json()
    print(json_)
    essay = pd.DataFrame(json_)
    jsons = json.dumps(json_)
    essay_new = json.loads(jsons)
    f1 = extract_features(essay)
    X = f1.iloc[:, 1:].values
    create_gensim_lsa_model(essay_new, 1, 5)
    prediction = clf.predict(X)
    prediction=np.squeeze(prediction.tolist())
    print('SVM:', prediction)
    global random_prompt
    vector1 = text_to_vector(str(preprocess_data(random_prompt)))
    vector2 = text_to_vector(str(preprocess_data(essay_new)))
    cosine = get_cosine(vector1, vector2)
    print('Cosine:', cosine)
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(jsons)
    
    print(len(matches))
    del matches[:3]
    del matches[len(matches)-1]
    final_prediction = cosine * 4 + prediction * 2 - len(matches) * 0.2
    print('Final:', final_prediction)
    value = ""
    if final_prediction>5:
        value = "Excellent"
    elif final_prediction>4:
        value = "Very Good"
    elif final_prediction>3:
        value="Good"
    elif final_prediction>2:
        value="Not Bad"
    else:
        value="Need Improvement"
    if len(matches)>0:
        return jsonify({'prediction': str(value),
                    'charcount': str(char_count.chars),
                    'wordcount': str(word_count.word),
                    'sentcount': str(sent_count.sent),
                    'avglen': str(avg_word_len.avrg), 
                    #'lemmacount': str(count_lemmas.lemma_count),
                    'spell': str(count_spell_error.mispell_count),
                    'punctuation': str(punctuation_count.p),
                    'nouncount': str(count_pos.noun_count),
                    'adjcount': str(count_pos.adj_count),
                    'verbcount': str(count_pos.verb_count),
                    'advcount': str(count_pos.adv_count),
                    'pronouncount': str(count_pos.pronoun_count),
                    'preposition': str(count_pos.preposition_count),
                    'matches' : "Number of Grammatical Errors :"+str(len(matches)),
                    'grammar': str(matches[0])
                    })
    else:
        return jsonify({'prediction': str(value),
                    'charcount': str(char_count.chars),
                    'wordcount': str(word_count.word),
                    'sentcount': str(sent_count.sent),
                    'avglen': str(avg_word_len.avrg), 
                    #'lemmacount': str(count_lemmas.lemma_count),
                    'spell': str(count_spell_error.mispell_count),
                    'punctuation': str(punctuation_count.p),
                    'nouncount': str(count_pos.noun_count),
                    'adjcount': str(count_pos.adj_count),
                    'verbcount': str(count_pos.verb_count),
                    'advcount': str(count_pos.adv_count),
                    'pronouncount': str(count_pos.pronoun_count),
                    'preposition': str(count_pos.preposition_count),
                    'matches' : "Number of Grammatical Errors :"+str(len(matches))
                    })


@app.route('/', methods=['GET', 'POST'])
def enter():
    #dataframe = pd.read_csv('C:/Users/ASUS/Documents/VSCode/AES_SVM/prompt.csv', encoding='latin-1')
    #random_prompt = dataframe.sample(n=1)
    #print(random_prompt)
    with open('C:/Users/ASUS/Documents/VSCode/AES_SVM/prompts.csv') as f:
        readers = csv.reader(f)
        global random_prompt
        random_prompt = random.choice(list(readers))
    return render_template('form.html', random_prompt = random_prompt)


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        port = 5000
    clf = joblib.load("model1.pkl")
    app.run(port=port, debug=True)
