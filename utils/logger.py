from datetime import datetime


class Logger:
    f = None

    def __init__(self, filename=None):
        if filename is not None:
            self.f = open(filename, 'a')

    def log(self, message):
        printed_msg = "[" + datetime.now().__str__() + "]: " + message
        print(printed_msg)
        if self.f is not None:
            self.f.write(printed_msg + '\n')
            self.f.flush()


