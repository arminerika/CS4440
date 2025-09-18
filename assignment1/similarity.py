# -------------------------------------------------------------------------
# AUTHOR: Armin Erika Polanco
# FILENAME: similarity.py
# SPECIFICATION: Find the two most similar documents by cosine similarity
# FOR: CS 4440 (Data Mining) - Assignment #1
# TIME SPENT: ~30 minutes
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy or pandas.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection using the white space as your character delimiter.
#--> add your Python code here

# Build the vocabulary
vocab = []
for doc in documents:
    text = doc[1].lower().split()  # lowercase and split on whitespace
    for word in text:
        if word not in vocab:      # check if the word is new
            vocab.append(word)     # add it to the vocabulary

# Build the binary document-term matrix
docTermMatrix = []
for doc in documents:
    words = doc[1].lower().split()    # tokenize current doc
    vector = [0] * len(vocab)         # initialize a vector of 0’s
    for w in words:
        if w in vocab:                # check if the word is in the vocabulary
            idx = vocab.index(w)      # find its index
            vector[idx] = 1           # mark presence with 1
    docTermMatrix.append(vector)      # store the vector for this doc

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here

best_sim = -1
best_pair = (None, None)

for i in range(len(docTermMatrix)):
    for j in range(i + 1, len(docTermMatrix)):
        # sklearn expects each vector wrapped in a list, since it works with 2D arrays
        sim = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]
        if sim > best_sim:
            best_sim = sim
            best_pair = (i, j)

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here

doc_i = documents[best_pair[0]][0]  # Document_ID of first doc in best pair
doc_j = documents[best_pair[1]][0]  # Document_ID of second doc
print(f"The most similar documents are document {doc_i} and document {doc_j} with cosine similarity = {best_sim:.4f}")