from sklearn import model_selection

class SampleSelector:
    dataset = None

    def __init__(self, dataset):
        self.dataset=dataset

    def strategy_25min(self):
        return [], []