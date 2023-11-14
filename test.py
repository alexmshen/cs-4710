import numpy as np
import pandas as pd
import tensorflow as tf
import backtrader as bt
import yfinance as yf
from random import sample

# Define the trading strategy
class DQNStrategy(bt.Strategy):
    def __init__(self):
        self.data0 = self.datas[0]
        self.data1 = self.datas[1]
        self.state_shape = (3,)
        self.nb_actions = 2
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, input_shape=self.state_shape, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.nb_actions, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
              loss='mse', metrics=['mae'])
        return model

    def _act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nb_actions)
        else:
            return np.argmax(self.model.predict(np.array([state])))

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array(state), action, reward, np.array(next_state), done))
        
    def _replay(self, batch_size, num_iterations):
        if len(self.memory) < batch_size:
            return

        minibatch = sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        for i in range(num_iterations):
            # Update the target model
            targets = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1) * (1 - dones)
            target_f = self.model.predict(states)
            target_f[np.arange(batch_size), actions] = targets
            self.model.fit(states, target_f, epochs=1, verbose=0)

        # Update the epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def next(self):
        # Calculate the moving average
        ma_period = 20
        close0 = self.data0.close.get(size=ma_period)
        close1 = self.data1.close.get(size=ma_period)

        ma0 = np.mean(close0)
        ma1 = np.mean(close1)

        # Calculate the z-score
        zscore0 = (self.data0.close[0] - ma0) / np.std(self.data0.close.get(size=ma_period), ddof=1)
        zscore1 = (self.data1.close[0] - ma1) / np.std(self.data1.close.get(size=ma_period), ddof=1)

        # Update the state
        state = np.array([zscore0, zscore1, self.data0.close[0] / self.data1.close[0]])

        # Get the action from the agent
        action = self._act(state)

        # Execute the action based on the DQN strategy
        if action == 0:
            self.buy(data=self.data0, size=20)
            self.sell(data=self.data1, size=20)
        else:
            self.sell(data=self.data0, size=20)
            self.buy(data=self.data1, size=20)

        # Check if the two stocks converge
        if abs(self.data0.close[0] - self.data1.close[0]) < 0.01 * (self.data0.close[0] + self.data1.close[0]) / 2:
            # Reverse the trade
            if action == 0:
                self.sell(data=self.data0, size=20)
                self.buy(data=self.data1, size=20)
            else:
                self.buy(data=self.data0, size=20)
                self.sell(data=self.data1, size=20)

        # Calculate the reward
        reward = self.broker.getvalue()

        # Update the memory
        next_state = np.array([zscore0, zscore1, self.data0.close[0] / self.data1.close[0]])
        done = False
        self._remember(state, action, reward, next_state, done)

        # Update the agent
        self._replay(batch_size=32, num_iterations=10)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
# Create the cerebro engine
cerebro = bt.Cerebro()

# Add the data feeds
data1 = bt.feeds.PandasData(dataname=yf.download('COKE', '2023-08-08', '2023-11-09'))
data2 = bt.feeds.PandasData(dataname=yf.download('PEP', '2023-08-08', '2023-11-09'))

cerebro.adddata(data1)
cerebro.adddata(data2)



# Add the strategy
cerebro.addstrategy(DQNStrategy)

# Run the backtest
cerebro.run()
cerebro.plot()