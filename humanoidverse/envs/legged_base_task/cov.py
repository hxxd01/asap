import numpy as np

class OnlineMeanCovBatch:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros((dim, dim))

    def update(self, batch):
        # batch: (batch_size, dim)
        batch = np.asarray(batch)
        m = batch.shape[0]
        if m == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_delta = batch - batch_mean
        batch_M2 = batch_delta.T @ batch_delta  # (dim, dim)

        if self.n == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n = m
        else:
            delta = batch_mean - self.mean
            new_n = self.n + m
            self.mean = (self.n * self.mean + m * batch_mean) / new_n
            self.M2 = self.M2 + batch_M2 + np.outer(delta, delta) * self.n * m / new_n
            self.n = new_n

    def get_mean(self):
        return self.mean

    def get_cov(self):
        if self.n < 2:
            return np.zeros_like(self.M2)
        return self.M2 / (self.n - 1)

    def save(self, filename):
        np.savez(filename, n=self.n, mean=self.mean, M2=self.M2)

    def load(self, filename):
        data = np.load(filename)
        self.n = int(data['n'])
        self.mean = data['mean']
        self.M2 = data['M2']