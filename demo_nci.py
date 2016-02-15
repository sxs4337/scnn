import numpy as np
from scnn import GraphSCNN, data
from sklearn.metrics import f1_score

# Parse the nci dataset and return an lists of adjacency matrices, list of design matrices, and a 1-hot label matrix
A, X, Y = data.parse_nci()

# Construct array indices for the training, validation, and test sets
n_nodes = len(A)
indices = np.arange(n_nodes)
train_indices = indices[:n_nodes // 3]
valid_indices = indices[n_nodes // 3:(2* n_nodes) // 3]
test_indices  = indices[(2* n_nodes) // 3:]

# Instantiate an SCNN and fit it to nci
scnn = GraphSCNN()
scnn.fit(A, X, Y, train_indices=train_indices, valid_indices=valid_indices)

# Predict labels for the test set 
preds = scnn.predict(A, X, test_indices)
actuals = np.argmax(Y[test_indices,:], axis=1)

# Display performance
print 'F score: %.4f' % (f1_score(actuals, preds))
