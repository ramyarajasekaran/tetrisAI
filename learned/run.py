from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Run dqn with Tetris
def dqn():
    episodes = 10000
    max_steps = None
    epsilon_stop_episode = 7000
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 1000
    log_every = 20
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    env = Tetris()

    '''
    with open(r"saved_agents/pickled_new_agent_10000_7000", "rb") as input_file:
        agent = pickle.load(input_file)
        agent.epsilon = 0
    
    '''
    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)
    agent.epsilon = 0
    '''
    hateris = DQNAgent(env.get_state_size()+1,
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)
    #env.hater = hateris
    '''

  
    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    tot_max_score = 0
    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0


        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states(env.current_piece)
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)
            #agent.add_to_memory(current_state, next_states[best_action], reward, done)
            #hateris.add_to_memory(current_state+[env.current_piece], next_states[best_action]+[env.current_piece], -reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        #if episode % train_every == 0:
            #agent.train(batch_size=batch_size, epochs=epochs)
            #hateris.train(batch_size=batch_size, epochs=epochs)

        # Logs
        #if log_every and episode and episode % log_every == 0 and episode>101:
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])
            print(str(episode) + " " + str(avg_score) +" "+  str(min_score)+ " "+
                    str(max_score))
            '''if (tot_max_score < max_score):
                agent.save("dqnAgentMax10000.h5", episode)
                tot_max_score = max_score'''

    #agent.save("dqnAgent10000.h5", episode)
   # with open("saved_agents/pickled_new_agent_10000_7000", "wb") as input_file:
        #pickle.dump(agent,input_file)
    plt.plot(scores)
    plt.show()

if __name__ == "__main__":
    dqn()
