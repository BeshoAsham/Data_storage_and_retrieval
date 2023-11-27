import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from natsort import natsorted
import math
import numpy as np
import pandas as pd

# Read files name
files_name = natsorted(os.listdir('DocumentCollection'))

# Download NLTK resources (you only need to do this once)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Create a PorterStemmer instance
porter = PorterStemmer()

# Function to take a document and return a list of terms after preprocessing
def preprocessing(doc):
    token_docs = word_tokenize(doc)

    prepared_doc = []
    for term in token_docs:
        # Remove stop words
        if term not in stop_words:
            # Apply stemming
            stemmed_term = porter.stem(term)
            prepared_doc.append(stemmed_term)
    return prepared_doc

document_of_terms = []
for files in files_name:
    with open(f'DocumentCollection\{files}', 'r') as f:
        document = f.read()
    document_of_terms.append(preprocessing(document))

print('Terms after tokenization, removing stop words, and stemming:')
print(document_of_terms)

# Ensure you apply stemming in the query_input function as well.

####### positional index #########
document_number = 0
positional_index = {}
for document in document_of_terms:
    # For position and term in the tokens.
    for positional, term in enumerate(document):
        # print(pos, '-->' ,term)
        # If term already exists in the positional index dictionary.
        if term in positional_index:

            # Increment total freq by 1.
            positional_index[term][0] = positional_index[term][0] + 1

            # Check if the term has existed in that DocID before.
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)

            else:
                positional_index[term][1][document_number] = [positional]

        # If term does not exist in the positional index dictionary
        # (first encounter).
        else:

            # Initialize the list.
            positional_index[term] = []
            # The total frequency is 1.
            positional_index[term].append(1)
            # The postings list is initially empty.
            positional_index[term].append({})
            # Add doc ID to postings list.
            positional_index[term][1][document_number] = [positional]

    # Increment the file no. counter for document ID mapping
    document_number += 1

print('Positional index')
print(positional_index)

### phrase Query ###
query = input('Input Phrase Query: ')

def query_input(q):
    lis = [[] for _ in range(10)]
    for term in preprocessing(q):
        if term in positional_index.keys():
            for key in positional_index[term][1].keys():
                if lis[key] != []:
                    # Check if the current position is consecutive to the last one
                    if lis[key][-1] + 1 == positional_index[term][1][key][0]:
                        lis[key].append(positional_index[term][1][key][0])
                    else:
                        # If not consecutive, start a new sequence
                        lis[key] = [positional_index[term][1][key][0]]
                else:
                    # If the list is empty, add the first position
                    lis[key].append(positional_index[term][1][key][0])

    positions = []
    for pos, lst in enumerate(lis, start=1):
        if len(lst) == len(preprocessing(q)):
            positions.append(f'doc{pos}')

    return positions
print(query_input(query))

######### Print Tabels before input Query #############
all_words = []
for doc in document_of_terms:
    for word in doc:
        all_words.append(word)

def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


term_freq = pd.DataFrame(get_term_freq(document_of_terms[0]).values(), index=get_term_freq(document_of_terms[0]).keys())

for i in range(1, len(document_of_terms)):
    term_freq[i] = get_term_freq(document_of_terms[i]).values()

term_freq.columns = ['doc ' + str(i) for i in range (1,11)]
print('TF')
print(term_freq)


def get_weighted_term_freq(x):
    if x > 0:
        return math.log10(x) + 1
    return 0


for i in range(1, len(document_of_terms) + 1):
    term_freq['doc ' + str(i)] = term_freq['doc ' + str(i)].apply(get_weighted_term_freq)

print('Weighted TF')
print(term_freq)


tfd = pd.DataFrame(columns=['freq', 'idf'])

for i in range(len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()

    tfd.loc[i, 'freq'] = frequency

    tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))

tfd.index = term_freq.index

print('IDF')
print(tfd)

term_freq_inve_doc_freq = term_freq.multiply(tfd['idf'], axis=0)

print('TF.IDF')
print(term_freq_inve_doc_freq)


document_length = pd.DataFrame()
def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x ** 2).sum())
for column in term_freq_inve_doc_freq.columns:
    document_length.loc[0, column + '_len'] = get_docs_length(column)

print('Document Length')
print(document_length)

normalized_term_freq_idf = pd.DataFrame()


def get_normalized(col, x):
    try:
        return x / document_length[col + '_len'].values[0]
    except:
        return 0


for column in term_freq_inve_doc_freq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_freq[column].apply(lambda x: get_normalized(column, x))

print('Nomalized TF.IDF')
print(normalized_term_freq_idf)


######## input Query ##########

def insert_query(q):

        query = pd.DataFrame(index=normalized_term_freq_idf.index)
        query['tf'] = [1 if x in preprocessing(q) else 0 for x in list(normalized_term_freq_idf.index)]
        query['w_tf'] = query['tf'].apply(lambda x: get_weighted_term_freq(x))
        product = normalized_term_freq_idf.multiply(query['w_tf'], axis=0)
        query['idf'] = tfd['idf'] * query['w_tf']
        query['tf_idf'] = query['w_tf'] * query['idf']
        query['normalized'] = 0
        for i in range(len(query)):
            query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values ** 2))
        print('Query Details')
        print(query.loc[preprocessing(q)])
        product2 = product.multiply(query['normalized'], axis=0)
        scores = {}
        for col in product2.columns:
            if 0 in product2[col].loc[preprocessing(q)].values:
                pass
            else:
                scores[col] = product2[col].sum()
        product_result = product2[list(scores.keys())].loc[preprocessing(q)]
        print()
        print('Product (query*matched doc)')
        print(product_result)
        print()
        print('product sum')
        print(product_result.sum())
        print()
        print('Query Length')
        q_len = math.sqrt(sum([x ** 2 for x in query['idf'].loc[preprocessing(q)]]))
        print(q_len)
        print()
        print('Cosine Simliarity')
        print(product_result.sum())
        print()
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print('Returned docs')
        for typle in sorted_scores:
            print(typle[0])


q = input('Input Query for print Query details and matched document')
insert_query(q)