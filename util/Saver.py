import os

class Saver():
    def __init__(self, path, mode = 'a'):
        self.file = open(path, mode)
        self.counter = 0
    def close(self):
        self.file.flush()
        self.file.close()
    def header(self, list):
        string = ''
        for i in list:
            string += (',' + str(i))
        string += os.linesep
        self.file.write(string)

    def save_1D(self, list):
        string = str(self.counter)
        for i in list:
            string += (','+ str(i))
        string += os.linesep
        self.file.write(string)
        self.counter +=1