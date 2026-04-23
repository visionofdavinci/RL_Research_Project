import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque
from collections import defaultdict
import matplotlib.pyplot as plt
from test_script import QNetwork
from test_script import bar_plot, test_pole_length, test_script
import matplotlib.pyplot as plt

class ReplayBuffer:
    """
    Replay Buffer to store experience tuples for deep q learning.
    The replay buffer stores experiences from many episodes and randomly samples them during training.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class StratifiedReplayBuffer:
    """
    Replay Buffer to store experience tuples for deep q learning.
    The replay buffer stores experiences from many episodes and randomly samples them during training.
    """
    def __init__(self, capacity):
        self.buffer = {}
        self.capacity = capacity
    
    def push(self, label, state, action, reward, next_state, done):
        sample = (state, action, reward, next_state, done)
        if label not in  self.buffer:
            self.buffer[label] = deque()
            num_labels = len(self.buffer)
            new_max_buffer_capacity = self.capacity // num_labels
            remainder = self.capacity % num_labels
            # dynamically adapt deque max cap as we add sampled pole lengths to our superbuffer. 
            for i, existing_label in enumerate(self.buffer.keys()):
                current_maxlen = new_max_buffer_capacity + (1 if i < remainder else 0)        
                old_deque = self.buffer[existing_label]
                self.buffer[existing_label] = deque(old_deque, maxlen=current_maxlen) 


        self.buffer[label].append(sample)

    def sample(self, batch_size):
        if not self.buffer:
            return []
        label_buffer = self.buffer[label]
        labels = list(self.buffer.keys()) 
        samples_per_label = batch_size//len(labels)
        remainder = batch_size % len(labels)
        all_samples = []
        # iterate through all labels (pole lengths) to ensure most even distribution in sample (uniform)
        for i, label in enumerate(labels):
            num_to_sample = samples_per_label + (1 if i < remainder else 0)
            num_to_sample = min(num_to_sample, len(label_buffer))
            batch = random.sample(label_buffer, num_to_sample)
            all_samples.extend(batch)
        
        random.shuffle(all_samples)
        return all_samples
    
    def __len__(self):
        return sum(len(d) for d in self.buffer.values())
    
class AdaptiveCurriculumLearning:
    """ 
    Adaptive curriculum learning offers an object to store the accumulated rewards, 
    performances, difficulty socores, and probabilities for each pole length. 
    We check the last n (LOOK_BACK_WINDOW) rewards a pole has obtained and assign probabilities 
    to sample each length for an episode. The goal is to attack the policy on its weak spots, 
    we prioritize training on weak performing lengths. This should shift performance metrics and keeps 
    our attacks adaptive as we target weakest. 
    param: all_pole_lengths : numpy.ndarray - stores all the pole lengths
    attr: p_adaptive : float - probability to select pole length from sample distribution, after uniform_episode_training_cap
    attr: lb_window : int - look back window used to compute performance metric, number of recent pole length performances    
    """
    LOOK_BACK_WINDOW = 20 # we only consider the last 20 rewards a pole has gotten (captures more sensitive information)
    P_ADAPTIVE = 0.9 # probability of using adaptive probability distribution or uniform random
    def __init__(self, all_pole_lengths):
        self.all_pole_lengths = all_pole_lengths
        self.rewards = defaultdict(list) # storing pole length episode reward for approach 2 (see 2.2)
        self.performances = {} # keep track of performance metric for each pole length 
        self.difficulty_scores = {} # difficulty_scores
        self.distribution = {} # probability distribution for pole lengths to sample from
        
        # initialize difficulties and probabilities (initially prob is uniform)
        initial_prob = 1.0 / len(all_pole_lengths)
        self.i_p = initial_prob
        for length in all_pole_lengths:
            self.difficulty_scores[length] = 1.0
            self.distribution[length] = initial_prob

    def update_rewards(self, pole_length, reward):
        self.rewards[pole_length].append(reward)

    def update_performances(self, pole_length):
        """
        Here we update the performance of a pole, 
        """
        reward_list = self.rewards[pole_length]

        # if a pole has not been played yet, it will be assigned a performance metric of 0
        if not reward_list:
            metric = 0
        else:
            # here we use a LOOK BACK WINDOW so that we dont use over-stabilized episode rewards
            metric = np.mean(reward_list[-self.LOOK_BACK_WINDOW:])
        self.performances[pole_length] = metric

    def update_difficulties(self):
        """
        Difficulties are inversely proportional to the performance metrics.
        The worse a pole length has performed, the higher the diff score (diff=1 being the worst performing, diff=0 being best).
        These require global updates as the update is relative to the totality of pole lengths.
        Also normalizing + scaling the values down. Metrics can offer quite larger values otherwise. 
        """
        M_max = self.find_max()
        M_min = self.find_min()
        diff_M = M_max - M_min

        # update all difficulty scores with new information
        for pole_length, metric in self.performances.items():
            if diff_M == 0:
                difficulty = 1
            else:
                difficulty = 1 - ((metric-M_min) / diff_M) # if best perform, diff is 0 >>> probability assignment will be 0
            self.difficulty_scores[pole_length] = difficulty
    
    def update_distribution(self):
        """ 
        Distribution is also global, here we require an update proportional to the totality. 
        Normalizing the probabilities (between 0 and 1)
        """
        if not self.difficulty_scores:
            return
        total_difficulty = sum(self.difficulty_scores.values())

        if total_difficulty > 0:
            for pole_length, difficulty in self.difficulty_scores.items():
                self.distribution[pole_length] = difficulty / total_difficulty
        else:
            for pole_length in self.all_pole_lengths:
                self.distribution[pole_length] = self.i_p
    
    def sample_length(self):
        """
        Here select the pole length using either uniform or categorical prob distribution. 
        """
        if random.random() < self.P_ADAPTIVE:
            pole_lengths = list(self.distribution.keys())
            probs = list(self.distribution.values())

            # if our prob dist is empty, we fallback to uniform selection
            if not pole_lengths or sum(probs) == 0:
                return np.random.choice(self.all_pole_lengths)
            
            # selection based on categorical sampling (discrete prob distribution)
            return np.random.choice(a=pole_lengths, p=probs, size=1)[0]
        else:
            return np.random.choice(self.all_pole_lengths)

    def find_max(self):
        """
        Find max performing pole length
        """
        if not self.performances:
            return 0
        return max(self.performances.values())

    def find_min(self):
        """
        Find min performing pole length
        """
        if not self.performances:
            return 0
        return min(self.performances.values())
    
    def calculate_pole_stats(self):
        """
        Nice display function for avg reward of each pole. Not functionally important at all. 
        """
        avg_pole_stats = {}
        
        for pole_length, rewards_list in self.rewards.items():
            # Only process if the list is not empty
            if rewards_list:
                avg_reward = np.mean(rewards_list)
                count = len(rewards_list)
            else:
                avg_reward = 0.0
                count = 0
            avg_pole_stats[pole_length] = {
                "average_reward": avg_reward,
                "episode_count": count
            }
        for p, avg in avg_pole_stats.items():
            print(f"Pole length {p} has avg reward {avg}")
        return avg_pole_stats
    

def select_action(state, policy_net, epsilon, action_dim):
    """
    Select action using epsilon-greedy policy - did it with epsilon-greedy because of Assignent 1
    """
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()


def deep_q_learning(epsilon, gamma, alpha, q_network, n_episodes,
                    uniform_episode_training_cap=None,
                    pole_lengths=None, env_name='CartPole-v1',
                    batch_size=64, buffer_capacity=50000, 
                    update_target_every=10, epsilon_decay=0.995, 
                    epsilon_min=0.01):
    """
    Deep q learning agent for CartPole-v1 environment with varying pole lengths.
    
    param: epsilon : float - initial exploration rate
    param: gamma : float - discount factor
    param: alpha : float - learning rate
    param: q_network : QNetwork or None - pre-initialized network or None to create new one
    param: n_episodes : int - number of training episodes
    param: uniform_episode_training_cap : Union[int, None] - number episodes trained with uniform length selection
    if None, no adaptive curriculum learning enabled
    param: pole_lengths : array-like or None - array of pole lengths to train on (default: linspace(0.4, 1.8, 30))
    param: env_name : str - gym environment name
    param: batch_size : int - batch size for training
    param: buffer_capacity : int - replay buffer capacity
    param: update_target_every : int - how often to update target network
    param: epsilon_decay : float - epsilon decay rate per episode
    param: epsilon_min : float - minimum epsilon value

    return: tuple : (policy_net, target_net, episode_returns)
        - trained networks and list of episode rewards
    """
    
    # initialization of environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # initialization of networks if not provided
    if q_network is None:
        policy_net = QNetwork(state_dim, action_dim)
        target_net = QNetwork(state_dim, action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    else:
        policy_net = q_network
        target_net = QNetwork(state_dim, action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

    # initialization of optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)

    # initialization of replay buffer
    replay_buffer = StratifiedReplayBuffer(buffer_capacity)
    
    # pole lengths for training
    if pole_lengths is None:
        pole_lengths = np.linspace(0.4, 1.8, 30)

    # storing episode returns for plotting
    episode_returns = []

    # initialize the acl class for adaptive hyperparametre sampling
    if uniform_episode_training_cap is not None:
        acl = AdaptiveCurriculumLearning(pole_lengths)
    else:
        acl = None

    # copy of current epsilon value for decay
    epsi = epsilon
    
    # training loop
    for episode in range(n_episodes):
        # if current episode is below the uniform_episode_training_cap we select from a uniform distr
        if uniform_episode_training_cap is None:
            pole_length = np.random.choice(pole_lengths)
        elif episode <= uniform_episode_training_cap: # or if acl not enabled
            pole_length = np.random.choice(pole_lengths)
        # else use adaptive probability distrubition 
        else: 
            pole_length = acl.sample_length()

        env.unwrapped.length = pole_length
        
        # reset environment
        state = env.reset()[0]
        episode_reward = 0.0
        
        # epsilon decay
        if epsi > epsilon_min:
            epsi = max(epsilon_min, epsi * epsilon_decay)
        
        # episode loop (1 episode = 1 pole length)
        done = False
        
        while not done:
            # select action
            action = select_action(state, policy_net, epsi, action_dim)
            
            # take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store transition in replay buffer
            replay_buffer.push(pole_length, state, action, reward, next_state, float(done))

            # deep q learning update (using mini-batch from replay buffer)
            if len(replay_buffer) >= batch_size:
                # sample batch from replay buffer
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # convert to tensors 
                states_t = torch.FloatTensor(states)
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_t = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones).unsqueeze(1)

                #get current q values
                current_q = policy_net(states_t).gather(1, actions_t)
                
                # target values
                with torch.no_grad():
                    next_max = target_net(next_states_t).max(1)[0].unsqueeze(1)
                    td_target = rewards_t + gamma * next_max * (1 - dones_t)
                
                # loss calc
                loss = nn.MSELoss()(current_q, td_target)
                
                # backprop and optimize + gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            
            episode_reward += reward
            state = next_state
        
        
        #update target network periodically
        if episode % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        #store episode reward
        episode_returns.append(episode_reward)
        
        # update acl for rewards, performance metrics, scores, and probs
        # difficulty scores and probabilities are global updates, can be seen in the respective update methods
        if uniform_episode_training_cap is not None:
            acl.update_rewards(pole_length, episode_reward)
            if episode >= uniform_episode_training_cap:
                acl.update_performances(pole_length)
                acl.update_difficulties()
                acl.update_distribution()

        #only for seeing the progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
            print(f"Episode {episode}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Epsilon: {epsi:.3f} | "
                  )            
    
    env.close()
    
    return policy_net, target_net, episode_returns, acl, uniform_episode_training_cap

def plot_episode_rewards_averaged(episode_rewards, episode_cap=None, window_size=50):
    rewards_series = np.array(episode_rewards)
    
    weights = np.ones(window_size) / window_size
    averaged_rewards = np.convolve(rewards_series, weights, mode='valid')

    averaged_episodes = np.arange(window_size, len(episode_rewards) + 1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(averaged_episodes, averaged_rewards, 
             label=f'Rolling Average (Window = {window_size})', 
             color='b') 
    
    # check we are using acl training
    if episode_cap > 0 and episode_cap <= len(episode_rewards) and episode_cap is not None:
        plt.axvline(x=episode_cap, color='r', linestyle='--', 
                    label=f'Training Shift (Episode {episode_cap})')
    
    raw_episodes = range(1, len(episode_rewards) + 1)
    plt.plot(raw_episodes, episode_rewards, 
             label='Raw Episode Reward', 
             color='gray', 
             alpha=0.3)
    
    plt.xlabel('Episode')
    plt.ylabel(f'Reward (Avg over {window_size} Episodes)')
    plt.title(f'Episode Reward Rolling Average (Window {window_size})')
    
    plt.xlim(0, len(episode_rewards) + 1)
    
    plt.legend()
    plt.grid(True)
    
    plt.savefig('averaged_episode_rewards_plot.png')
    plt.close()
