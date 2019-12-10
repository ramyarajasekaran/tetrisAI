# tetrisAI

## Loading saved NN

in the dqn_agent.py file, replace the function _build_model with this:


        
    def _build_model(self):
        '''Builds a Keras deep neural network model
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        '''
        model = load_model('dqnAgent.h5')
        print("loaded dqnAgent")
        
        return model
## Loading pickled agent
in run.py file :


    with open(r"pickled_dqn", "rb") as input_file:
        agent = pickle.load(input_file)
    '''
    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)
    '''
   
## Using compute_avg
python3 compute_avg.py path/to/txtfile

like this:

        Prajaktas-MacBook-Pro:tetrisAI prajjoshi$ python3 compute_avg.py learned/saved_agents/pickled_new_agent_2800_1800_output
        avg: 11.022611940298505
        initial: 0.03
        final: 13.32


ur txt file should look something like this 


                wireless-10-145-207-190:learned prajjoshi$ python3 run.py
                Using TensorFlow backend.
                2019-12-10 15:15:54.682653: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that 
                this TensorFlow binary was not compiled to use: AVX2 FMA
                2019-12-10 15:15:54.694570: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f92ed441240 executing 
                computations on platform Host. Devices:
                2019-12-10 15:15:54.694591: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, 
                Default Version
                  4%|█▋                                      | 120/2800 [01:14<25:04,  1.78it/s]120 5.4 0 19
                  5%|██                                      | 140/2800 [01:27<28:00,  1.58it/s]140 5.57 0 22
                  6%|██▎                                     | 160/2800 [01:38<23:07,  1.90it/s]160 5.44 0 22
                  6%|██▌                                     | 180/2800 [01:50<24:35,  1.78it/s]180 5.81 0 22
                  7%|██▊                                     | 200/2800 [02:02<23:00,  1.88it/s]200 6.11 0 22
                  8%|███▏                                    | 220/2800 [02:13<28:27,  1.51it/s]220 6.23 0 22
                  9%|███▍                                    | 240/2800 [02:24<25:22,  1.68it/s]240 5.67 0 22
                  9%|███▋                                    | 260/2800 [02:35<25:21,  1.67it/s]260 6.21 0 22
                 10%|████                                    | 280/2800 [02:47<25:02,  1.68it/s]280 6.36 0 22
                 11%|████▎                                   | 300/2800 [02:59<26:31,  1.57it/s]300 5.9 0 18
                 11%|████▌                                   | 320/2800 [03:11<25:35,  1.62it/s]320 6.16 0 18
         ...

just make sure that the only % signs in ur file are in the lines with the bars
