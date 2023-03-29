class TextDBNScheduler:
    """Learning rate scheduler for TextDBN specified in the paper. 

    Learning rate is halved every 4000 steps that the optimizer takes.   
    """

    def __init__(self, optimizer):
        """Constructor of TextDBNScheduler.

        Args:
            optimizer: optimizer used for training.
        """

        self.optimizer = optimizer
        self._step = 0
        self.steps_for_change = 4000

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.

        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        self.__dict__.update(state_dict)

    def step(self):
        self._step += 1
        for p in self.optimizer.param_groups:
            if self._step % self.steps_for_change == 0:
                p['lr'] = p['lr'] / 2
