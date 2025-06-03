import numpy as np

class ocbc_dataset:
    def __init__(self, dataset, batch_size, gamma=0.99, frame_stack=None, action_len=None, p_aug=None):
        # Only support compact (observation-only) datasets.
        assert 'next_observations' not in dataset
        
        self.observation = dataset['observations']
        self.action = dataset['actions']

        self.valid_start_ids = np.nonzero(dataset['valids'] == 1)[0]  #indices which can be sampled as states for OCBC
        terminal_locs = np.nonzero(dataset['valids'] == 0)[0]         #indices corresponding to terminal states
        initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
        indices = np.searchsorted(terminal_locs, np.arange(len(dataset['observations'])), side='left')
        self.ends = terminal_locs[np.minimum(indices, len(terminal_locs) - 1)]
        
        if frame_stack is not None:
            initial_state_idxs = initial_locs[np.searchsorted(initial_locs, np.arange(len(dataset['observations'])), side='right') - 1]

            rets = []
            for i in reversed(range(frame_stack)):
                cur_idxs = np.maximum(np.arange(len(dataset['observations'])) - i, initial_state_idxs)
                rets.append(dataset['observations'][cur_idxs])
            self.observation = np.concatenate(rets, axis=-1)

        if action_len is not None:
            self.valid_start_ids = self.valid_start_ids[(self.ends[self.valid_start_ids] - self.valid_start_ids >= action_len)]
                    
        terminal_locs = indices = initial_locs = rets = None

        self.action_len = action_len
        self.batch_size = batch_size
        self.p_aug = p_aug
        self.size = len(dataset['observations'])
        self.gamma = gamma
        
    def sample(self, batch_size=None, evaluation=False):
        if batch_size is None: batch_size = self.batch_size
            
        current_idxs = self.valid_start_ids[np.random.randint(len(self.valid_start_ids), size=batch_size)]
        
        offsets = np.random.geometric(p=1 - self.gamma, size=batch_size)
        future_ids = np.minimum(current_idxs + offsets, self.ends[current_idxs])

        assert np.all(self.ends[future_ids] == self.ends[current_idxs])
        
        if self.p_aug is not None and not evaluation:
            raise NotImplementedError

        if self.action_len is not None:
            return self.observation[current_idxs], self.action[current_idxs[:, None] + np.arange(self.action_len)].reshape(self.batch_size, -1), self.observation[future_ids]
        else:
            return self.observation[current_idxs], self.action[current_idxs], self.observation[future_ids]