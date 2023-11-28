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
    encountered_terms = set()  # Keep track of encountered terms in the current document

    for positional, term in enumerate(document):
        # If term already exists in the positional index dictionary.
        if term in positional_index:

            # Check if the term has not been encountered in the current document yet.
            if term not in encountered_terms:
                # Increment the number of documents the term appears in.
                positional_index[term][0] = positional_index[term][0] + 1

                # Add the current document to the postings list.
                positional_index[term][1][document_number] = [positional]

                # Mark the term as encountered in the current document.
                encountered_terms.add(term)

            else:
                # If the term has already been encountered in the current document,
                # append the current position to the existing list.
                positional_index[term][1][document_number].append(positional)

        # If term does not exist in the positional index dictionary (first encounter).
        else:
            # Initialize the list.
            positional_index[term] = []
            # The total frequency is 1.
            positional_index[term].append(1)
            # The postings list is initially empty.
            positional_index[term].append({})
            # Add doc ID to postings list.
            positional_index[term][1][document_number] = [positional]

            # Mark the term as encountered in the current document.
            encountered_terms.add(term)

    # Increment the file no. counter for document ID mapping
    document_number += 1

print('Positional index')
print(positional_index)

### phrase Query ###
# query = input('Input Phrase Query: ')

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


# print(query_input(query))

######### Print Tables before input Query #############
all_words = []
for doc in document_of_terms:
    for word in doc:
        all_words.append(word)


def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


term_freq = pd.DataFrame(index=get_term_freq(document_of_terms[0]).keys())

for i in range(0, len(document_of_terms)):
    term_freq[i] = get_term_freq(document_of_terms[i]).values()

term_freq.columns = ['doc ' + str(i) for i in range(1, 11)]
print('TF')
print(term_freq)


##################################################################################################################
def get_weighted_term_freq(x):
    if x > 0:
        return math.log10(x) + 1
    return 0


weighted_term_freq = pd.DataFrame(index=get_term_freq(document_of_terms[0]).keys())

for i in range(1, len(document_of_terms) + 1):
    weighted_term_freq[i] = term_freq['doc ' + str(i)].apply(get_weighted_term_freq)
weighted_term_freq.columns = ['doc ' + str(i) for i in range(1, 11)]

print('Weighted TF')
print(weighted_term_freq)
######################################################################################################################
DF_IDF = pd.DataFrame(columns=['df', 'idf'])

for i in range(len(term_freq)):
    # Count the number of documents the term appears in (non-zero entries in the column).
    frequency = np.count_nonzero(term_freq.iloc[i].values)

    DF_IDF.loc[i, 'df'] = frequency

    DF_IDF.loc[i, 'idf'] = math.log10(len(document_of_terms) / (float(frequency)))

DF_IDF.index = term_freq.index
print('IDF')
print(DF_IDF)


term_freq_inve_doc_freq = term_freq.multiply(DF_IDF['idf'], axis=0)
print('TF.IDF')
print(term_freq_inve_doc_freq)

document_length = pd.DataFrame()

##############################################################################
def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x ** 2).sum())


for column in term_freq_inve_doc_freq.columns:
    document_length.loc[0, column + '_len'] = get_docs_length(column)

print('Document Length')
print(document_length)
###############################################################################################
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
# Function to get the total frequency of a term across all documents
def get_term_frequency(term):
    return term_freq.loc[term].sum()

def insert_query(q):
    terms_in_documents = set(normalized_term_freq_idf.index)

    # Tokenize, remove stop words, and stem the query terms
    query_terms = preprocessing(q)

    # Count the occurrences of each term in the query
    term_counts = {}
    for term in query_terms:
        term_counts[term] = term_counts.get(term, 0) + 1

    # Check if each query term exists in the documents
    for term, count in term_counts.items():
        if term in terms_in_documents:
            total_frequency = get_term_frequency(term)
            print(f'Term "{term}" exists in the query {count} times and in the documents {total_frequency} times.')
        else:
            print(f'Term "{term}" does not exist in the documents.')

    # Filter out non-existing terms from the query
    existing_query_terms = [term for term in term_counts.keys() if term in terms_in_documents]

    if not existing_query_terms:
        print('No valid terms found in the query. Exiting...')
        return

    # Further processing for existing query terms
    query = pd.DataFrame(index=normalized_term_freq_idf.index)
    query['tf'] = [term_counts.get(x, 0) for x in query.index]
    query['w_tf'] = query['tf'].apply(lambda x: get_weighted_term_freq(x))
    query['idf'] = DF_IDF['idf']
    query['tf_idf'] = query['w_tf'] * query['idf']
    query['normalized'] = query['tf_idf'] / np.sqrt((query['tf_idf'] ** 2).sum())

    print('Query Details:')
    print(query.loc[existing_query_terms])

    product2 = normalized_term_freq_idf.multiply(query['normalized'], axis=0)
    scores = {}
    for col in product2.columns:
        if 0 in product2[col].loc[existing_query_terms].values:
            pass
        else:
            scores[col] = product2[col].sum()

    if not scores:
        print('No matching documents found.')
        return

    product_result = product2[list(scores.keys())].loc[existing_query_terms]

    print('\nProduct (query * matched doc):')
    print(product_result)
    print('\nProduct sum:')
    print(product_result.sum())
    print('\nQuery Length:')
    q_len = math.sqrt(sum([x ** 2 for x in query['tf_idf'].loc[existing_query_terms]]))
    print(q_len)
    print('\nCosine Similarity:')
    print(product_result.sum())
    print('\nReturned docs:')

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for doc_id, score in sorted_scores:
        print(f'Doc {doc_id} - Score: {score}')


q = input('Input Query for print Query details and matched document: ')
insert_query(q)