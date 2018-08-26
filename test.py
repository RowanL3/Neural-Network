import random

from copy import deepcopy
from data import Database
from neural_network import Network, loss

# --- Deprecated ---
# Use ten_fold.py instead

nn = Network((4,4,5,3))

training_db = Database("iris_data_training.txt")
print("Training ({} samples): ".format(len(training_db.rows)))

num_epoch = 100
num_reports = 10
n = num_epoch / num_reports

min_loss = 5000
min_nn = nn
min_epoch = 0

for epoch in range(num_epoch):
    training_db.shuffle()
    results = list()
    for row in training_db.rows:
        row.output_vect = nn.forward(row.normalized)
        nn.backpropagate(row.normalized, row.output_vect, row.type_vect)
    total_loss = sum([loss(row.type_vect, row.output_vect) for row in training_db.rows])
    if(total_loss < min_loss):
        min_nn = deepcopy(nn)
        min_epoch = epoch
        min_loss = total_loss
    if (epoch % n == 0):
        print("  Epoch: {:03}, Training Loss: {:.3f} ".format(epoch, total_loss))

print("Lowest training loss:")
print("  Epoch: {:03}, Training Loss: {:.3f} ".format(min_epoch, min_loss))


testing_db = Database("iris_data_testing.txt")
print("Testing ({} samples): ".format(len(testing_db.rows)))

num_correct = 0
for row in testing_db.rows:
    output = min_nn.forward(row.normalized)
    prediction = [int(x == max(output)) for x in output]
    if prediction == row.type_vect:
        num_correct += 1

print("  {} / {} correctly identified".format(num_correct,len(testing_db.rows)))
