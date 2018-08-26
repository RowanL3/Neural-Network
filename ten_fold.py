from data import Database
from random import shuffle
from neural_network import Network, loss
from copy import deepcopy

db = Database("iris_data.txt")
nn = Network((4,5,3))

verbose = True
num_folds = 10
num_reports = 4

def n_fold(n):
    """Should return a list of tuples of training and testing data for every 
    run"""
    folds = list()
    training_size = len(db.rows) // n 
    for fold in range(n):
        start = training_size * fold
        end = training_size * fold + training_size
        testing = db.rows[start:end]
        training = db.rows[:start] + db.rows[end:]
        folds.append((training, testing))
    return folds

def run(num_epoch, training, testing):
    # Find a better place for this stuff
    epochs_per_report = num_epoch // num_reports
    min_loss = 5000
    min_nn = nn
    min_epoch = 0
    say("Training ({} samples): ".format(len(training)))
    for epoch in range(num_epoch):
        shuffle(training)
        results = list()
        for row in training:
            row.output_vect = nn.forward(row.normalized)
            nn.backpropagate(row.normalized, row.output_vect, row.type_vect)
        total_loss = sum([loss(row.type_vect, row.output_vect) for row in training])
        if(total_loss < min_loss):
            min_nn = deepcopy(nn)
            min_epoch = epoch
            min_loss = total_loss
        if (epoch % epochs_per_report == 0):
            say("  Epoch: {:03}, Training Loss: {:.3f} ".format(epoch, total_loss))

    say("Lowest training loss:")
    say("  Epoch: {:03}, Training Loss: {:.3f} ".format(min_epoch, min_loss))


    say("Testing ({} samples): ".format(len(testing)))

    num_correct = 0
    for row in testing:
        output = min_nn.forward(row.normalized)
        if row.matches(output):
            num_correct += 1
     
    say("  {} / {} correctly identified".format(num_correct,len(testing)))

    return num_correct

def say(str):
    if verbose == True:
        print(str)

if __name__ == "__main__":
    folds = n_fold(num_folds)
    total = sum([run(50, training, testing) for training, testing in folds])
    say("")
    print("Correctly identified {:.0%} of data in ".format(total/len(db.rows)))
