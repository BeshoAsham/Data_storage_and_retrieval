import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from natsort import natsorted
import math
import numpy as np
import pandas as pd
import re
# Read files name
files_name = natsorted(os.listdir('DocumentCollection'))


# nltk.download('punkt')

# Create a PorterStemmer instance
porter = PorterStemmer()


# Function to take a document and return a list of terms after preprocessing
def preprocessing(doc):
    token_docs = word_tokenize(doc)

    prepared_doc = []
    for term in token_docs:
        # Apply stemming
        stemmed_term = porter.stem(term)
        prepared_doc.append(stemmed_term)
    return prepared_doc


document_of_terms = []
for files in files_name:
    with open(f'DocumentCollection\{files}', 'r') as f:
        document = f.read()
    document_of_terms.append(preprocessing(document))

print('Terms after tokenization, and stemming:')
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

term_freq.columns = ['doc ' + str(i) for i in range(1, len(document_of_terms)+1)]
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
weighted_term_freq.columns = ['doc ' + str(i) for i in range(1, len(document_of_terms)+1)]

print('Weighted TF')
print(weighted_term_freq)
############################################### DF_IDF #######################################################################
DF_IDF = pd.DataFrame(columns=['df', 'idf'])

for i in range(len(term_freq)):
    # Count the number of documents the term appears in (non-zero entries in the column).
    frequency = np.count_nonzero(term_freq.iloc[i].values)

    DF_IDF.loc[i, 'df'] = frequency

    DF_IDF.loc[i, 'idf'] = math.log10(len(document_of_terms) / (float(frequency)))

DF_IDF.index = term_freq.index
print('IDF')
print(DF_IDF)
###################################### TF.IDF ######################################################
term_freq_inve_doc_freq = weighted_term_freq.multiply(DF_IDF['idf'], axis=0)
print('TF.IDF')
print(term_freq_inve_doc_freq)

##################################### Document Length #########################################
document_length = pd.DataFrame()
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


### phrase Query ###
def phrase_Query(q):
    lis = [[] for _ in range(len(document_of_terms))]
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
    #list of phrase query documents
    positions = []
    for pos, lst in enumerate(lis, start=1):
        if len(lst) == len(preprocessing(q)):
            positions.append(f'doc{pos}')

    return positions


def process_phrase_query(query):
    terms_in_documents = set(normalized_term_freq_idf.index)

    # Tokenize, and stem the query terms
    query_terms = preprocessing(query)

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
            print(f'Term "{term}" exists in the query {count} times and does not exist in the documents.')

    # Filter out non-existing terms from the query
    existing_query_terms = [term for term in term_counts.keys() if term in terms_in_documents]

    if not existing_query_terms:
        print(f'No valid terms found in the phrase query: {query}')
        return set()  # Return an empty set if no valid terms are found

    # Handling normalized values
    query_details = pd.DataFrame(index=term_counts, columns=['tf', 'w_tf', 'idf', 'tf_idf', 'normalized'])
    for term in query_terms:
        query_details.at[term, 'tf'] = term_counts.get(term, 0)
        query_details.at[term, 'w_tf'] = get_weighted_term_freq(query_details.at[term, 'tf'])
        query_details.at[term, 'idf'] = DF_IDF['idf'].get(term, 0)
        query_details.at[term, 'tf_idf'] = query_details.at[term, 'w_tf'] * query_details.at[term, 'idf']

    if (query_details['tf_idf'] ** 2).sum() != 0:
        query_details['normalized'] = query_details['tf_idf'] / np.sqrt((query_details['tf_idf'] ** 2).sum())
    else:
        # Handle the case where the denominator is zero (e.g., set 'normalized' to NaN)
        query_details['normalized'] = np.nan

    print('Query Details:')
    print(query_details)

    # Check if there are matched documents before proceeding with the cosine similarity calculation
    positions = phrase_Query(query)
    if not positions:
        print(f'No matching documents found for the phrase query: {query}')
        return set()  # Return an empty set if no matching documents are found

    product2 = normalized_term_freq_idf.multiply(query_details['normalized'], axis=0)
    scores = {}
    for col in product2.columns:
        if 0 in product2[col].loc[existing_query_terms].values:
            pass
        else:
            scores[col] = product2[col].sum()

    product_result = product2[list(scores.keys())].loc[existing_query_terms]

    print('\nProduct (query * matched doc):')
    print(product_result)
    print('\nCosine Similarity:')
    print(product_result.sum())
    print('\nQuery Length:')
    q_len = math.sqrt(sum([x ** 2 for x in query_details['tf_idf'].loc[existing_query_terms]]))
    print(q_len)
    print('\nReturned docs:')

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for doc_id, score in sorted_scores:
        print(f'Doc {doc_id} - Score: {score}')

    return positions  # Return the document positions


def process_boolean_query(boolean_query):
    # Split the boolean query into sub-queries based on AND, OR, and NOT operators
    sub_queries = re.split(r'\bAND\b|\bOR\b|\bNOT\b', boolean_query, flags=re.IGNORECASE)
    sub_queries = [sub_query.strip() for sub_query in sub_queries if sub_query.strip()]

    # Accumulate results for each sub-query individually
    results = []
    for sub_query in sub_queries:
        result = process_phrase_query(sub_query)
        results.append(result)

    # Apply boolean operators to combine results at the end
    combined_results = set(results[0])
    operator_index = 1
    for operator in re.finditer(r'\bAND\b|\bOR\b|\bNOT\b', boolean_query, flags=re.IGNORECASE):
        current_operator = operator.group().upper()
        result = set(results[operator_index])
        if current_operator == 'AND':
            combined_results &= result
        elif current_operator == 'OR':
            combined_results |= result
        elif current_operator == 'NOT':
            combined_results -= result
        operator_index += 1

    return combined_results

def insert_query(q):
    # Check if the query contains boolean operators
    if re.search(r'\bAND\b|\bOR\b|\bNOT\b', q, flags=re.IGNORECASE):
        results = process_boolean_query(q)
        print('Final Result:', sorted(list(results)))
    else:
        # If there are no boolean operators, process a single phrase query
        process_phrase_query(q)

# Example usage
q = input('Input Phrase Query (Boolean operators are supported) : ')
insert_query(q)