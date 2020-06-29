
import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = 256
        hid2_size = 64

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='softmax')

    def forward(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        out = self.fc3(h2)
        return out
