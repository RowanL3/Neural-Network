from operator import itemgetter
from random import shuffle

class Row():
    def __init__(self, text):
        self.cells = [float(x) for x in text.split(",")[:-1]]
        self.type = text.split(",")[-1].strip()
        self.type_vect = [int(x) for x in (self.type == "Iris-setosa",
                                            self.type == "Iris-versicolor",
                                            self.type == "Iris-virginica")]

class Data():
    def __init__(self):
        self.rows = list()
        self.col_max = list()
        self.col_min = list()

        with open("iris_data.txt") as d:
            for line in d.readlines():
                self.rows.append(Row(line))

        for i in range(4):
            self.col_min.append(min(self.get_collum(i)))
            self.col_max.append(max(self.get_collum(i)))

        for row in self.rows:
            row.normalized = [self.normalize_cell(i,cell) for (i,cell) in enumerate(row.cells)]
            
        self.shuffle()

    def shuffle(self):
        shuffle(self.rows)

    def normalize_cell(self,i,cell):
        min = self.col_min[i]
        max = self.col_max[i]
        return (cell-min)/(max-min)

    def get_collum(self,i):
        return map(lambda row: row.cells[i], self.rows)
