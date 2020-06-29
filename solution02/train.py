
import os
import gym
import numpy as np
import parl

from agent import Agent
from model import Model
from parl.algorithms import PolicyGradient

from parl.utils import logger

LEARNING_RATE = 1e-3


def convert_action(action):
    print(action)
    if 0 == action:
        return [ 0.0, 0.0, 0.0] # STRAIGHT
    elif 1 == action:
        return [ 1.0, 0.0, 0.0] # RIGHT
    elif 2 == action:
        return [-1.0, 0.0, 0.0] # LEFT
    elif 3 == action:
        return [ 0.0, 1.0, 0.0] # GO
    elif 4 == action:
        return [ 0.0, 0.0, 0.5] # BRAKE
    elif 5 == action:
        return [ 1.0, 0.0, 0.5] # BRAKE and RIGHT
    elif 6 == action:
        return [-1.0, 0.0, 0.5] # BRAKE and LEFT
    else:
        print("action error")

def warm_up(env):
    for i in range(15):
        action = (0, 1, 0)
        obs, reward, done, info = env.step(action)
        # env.render()
    for i in range(30):
        action = (0, 0, 0)
        obs, reward, done, info = env.step(action)
        # env.render()

def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    reward_cnt = 0

    while True:
        obs = preprocess(obs)  # from shape (96, 96, 3) to (28*32,)
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        action = convert_action(action)
        obs, reward, done, info = env.step(action)

        if reward < 0:
            reward_cnt += 1
        else:
            reward_cnt = 0
        if reward_cnt > 8:
            reward = -100
            done = True

        # print(action, reward) # for debug
        reward_list.append(reward)

        if done:
            reward_cnt = 0
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs)
            action = agent.predict(obs)
            action = convert_action(action)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def preprocess(image): # input 96*96*3 output 28*32
    image = image[:84]
    image = image[::3,::3,]

    # grass to black
    mask = np.all(image == [102, 229, 102], axis=2)
    image[mask] = [0, 0, 0]
    mask = np.all(image == [102, 204, 102], axis=2)
    image[mask] = [0, 0, 0]

    image = image[:, :, 0]
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色

    return image.astype(np.float).ravel()


def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


def main():
    env = gym.make('CarRacing-v0')
    obs_dim = 28 * 32
    act_dim = 7
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # 加载模型
    # if os.path.exists('./model.ckpt'):
    #     agent.restore('./model.ckpt')

    for i in range(3000):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            logger.info("Train Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(env, agent, render=False)
            logger.info('Episode {}, Test reward: {}'.format(
                i + 1, total_reward))
            ckpt = './model_episode_{}.ckpt'.format(i + 1)
            agent.save(ckpt)

    # save the parameters to ckpt
    agent.save('./model_final.ckpt')

    env.close()


if __name__ == '__main__':
    main()
