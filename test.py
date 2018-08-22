import random

from data import Database
from nn_ import Network, loss

nn = Network((4,5,3))

print("Training: ")
training_db = Database("iris_data_training.txt")

num_epoch = 1000
num_reports = 10
n = num_epoch / num_reports

for epoch in range(num_epoch):
    training_db.shuffle()
    results = list()
    for row in training_db.rows:
        row.output_vect = nn.forward(row.normalized)
        nn.backpropagate(row.normalized, row.output_vect, row.type_vect)
    total_loss = sum([loss(row.type_vect, row.output_vect) for row in training_db.rows])
    if (epoch % n == 0):
        print("Epoch: {:03}, Training Loss: {:.3f} ".format(epoch, total_loss))


print("Testing: ")
testing_db = Database("iris_data_testing.txt")

num_correct = 0
for row in testing_db.rows:
    output = nn.forward(row.normalized)
    l = [int(x == max(output)) for x in output]
    if l == row.type_vect:
        num_correct += 1

print("  {} / {} correctly identified".format(num_correct,len(testing_db.rows)))
