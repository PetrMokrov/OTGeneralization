import torch

class StatisticCollector:

    def __init__(self, n_averaging):
        self.n_averaging = n_averaging
        self.collected = 0
        self.history = []
        self.current = 0.

    
    def add(self, value):
        if self.collected < self.n_averaging:
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.current += value
            self.collected += 1

        if self.collected == self.n_averaging:
            self.history.append(self.current/self.n_averaging)
            self.reset_current()
    
    def reset_current(self):
        self.collected = 0
        self.current = 0.
    
    def reset(self):
        self.history = []
        self.reset_current()
    
    def get_history(self, inscribe_incomplete=False, incomplete_limit=1):
        history = self.history
        if inscribe_incomplete:
            assert(incomplete_limit >= 1)
            if self.collected >= incomplete_limit:
                history.append(self.current/self.collected)
        return history
    
    def plot_statistic(self, ax, title='Loss', xname='Iterations'):
        history = self.get_history()
        ax.plot(
            range(0, self.n_averaging * len(history), self.n_averaging),
            history
        )
        ax.set_title(title)
        ax.set_xlabel(xname)
