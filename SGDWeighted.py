import numpy as np 

class SGDWeighted:
    """ 
    Stochastic Gradient Descent by weighted averaging 
        constanst from paper:
        alpha  := step_alpha 
        c      := step_c
        M      := step_M
        beta   := weight_beta
        k_max  := num_iterations
        gamma  := step_scheduler
        omega  := weight_scheduler   
    """

    def __init__(self, params, grad, step_alpha, step_c, step_M, weight_beta, num_iterations, num_fncs, batch_size):
        self.iterate = params.copy()
        self.grad = grad
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.time_steps = np.arange(0, num_iterations + 1)
        self.step_scheduler = step_c * np.power(step_M / (self.time_steps + step_M), step_alpha)
        self.weight_scheduler = np.power(self.time_steps, weight_beta)
        self.num_fncs = np.arange(num_fncs)

        assert batch_size < num_fncs
    
    def optimize(self):
        """ Optimization algorithm """

        sum_weights = np.sum(self.weight_scheduler)
        cum_wighted_sum_iterates = 0

        for k in range(self.num_iterations):

            batch_idx = np.random.choice(self.num_fncs, self.batch_size)
            self.iterate -= self.step_scheduler[k] * self.grad(self.iterate, batch_idx)

            cum_wighted_sum_iterates += self.weight_scheduler[k+1] * self.iterate

        return cum_wighted_sum_iterates / sum_weights



