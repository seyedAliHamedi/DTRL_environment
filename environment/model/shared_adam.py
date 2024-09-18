import torch


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=0.005):
        super(SharedAdam, self).__init__(params, lr=lr)
                                        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor(0.0).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['step'] += 1  # Increment the step tensor
        super(SharedAdam, self).step(closure)
