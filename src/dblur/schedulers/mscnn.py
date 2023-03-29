class MSCNNScheduler:
    """
    MSCNN scheduler used for training a MSCNN model. 

    The learning rate scheduler ensures that the learning rate is the same for
    for a number of steps taken by the optimizer. After this threshold is
    reached, the learning rate is decreased by a certain factor.
    """

    def __init__(self, optimizer, steps_for_change=3e5, decrease_factor=0.1):
        """
        MSCNNScheduler Constructor.

        Args:
            steps_for_change: number of steps taken by optimizer before learning
                rate is modified.
            decrease_factor: factor by which learning rate is decreased after
                optimizer takes number of steps specified by steps_for_change.
        """

        self.optimizer = optimizer
        self.steps_for_change = steps_for_change
        self.decrease_factor = decrease_factor
        self._step = 0

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
            if self._step == self.steps_for_change:
                p['lr'] = (1 - self.decrease_factor) * p['lr']
