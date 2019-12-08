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
        print("loaded dqnAgentMax10000")
        
        return model
