import time


class Clock():
    def __init__(self):
        self.set(-1000000)

    def get(self):
        return self.pts_drift + time.time()

    def set(self, pts):
        # print(f'setting clock to {pts}')
        self.pts = pts
        self.pts_drift = pts - time.time()
