from collections import deque
from torch.utils.data import Dataset
import sys
import pickle


class ReplayBuffer:
    def __init__(self, size=sys.maxsize, capacity=1000000):
        self.size = size
        self.buf = deque(maxlen=capacity)

    def push(self, seq, energy):
        self.buf.append((seq, energy))
        if len(self.buf) > self.size:
            self.buf.popleft()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buf, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.buf = pickle.load(f)

    def __getitem__(self, idx):
        seq, energy = self.buf[idx]
        return {"idx": seq, "energy": energy}

    def __len__(self):
        return len(self.buf)


class BufferDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer, repetition):
        self.buffer = buffer
        self.repetition = repetition

    def __getitem__(self, idx):
        idx = idx % len(self.buffer)
        item = self.buffer[idx]
        return {"idx": item["idx"], "energy": item["energy"]}

    def __len__(self):
        return len(self.buffer) * self.repetition
