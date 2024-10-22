
import torch
import torch.nn as nn
import torch.optim as optim



##################### UTILITY CLASSES #####################

# Core Scheduler manages a forest of decision trees (SubDDTs)
class CoreScheduler(nn.Module):
    def __init__(self, subtree_input_dims, subtree_max_depth, devices, subtree_lr=0.005):
        super(CoreScheduler, self).__init__()
        
        # Initialize the decision trees for each device
        self.forest = [self.create_tree(subtree_input_dims + device['num_cores'], subtree_max_depth, device['num_cores']*3) for device in devices]
        self.optimizers = [optim.Adam(tree.parameters(), lr=subtree_lr) for tree in self.forest]

    # Create a decision tree (SubDDT) for the scheduler
    def create_tree(self, input_dims, subtree_max_depth, output_dim):
        return SubDDT(input_dims, output_dim, 0, subtree_max_depth)

# Decision Tree for the Actor (uses recursion to build a tree)
class DDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth):
        super(DDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth

        # Decision node if depth < max_depth
        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))

        # Leaf node if depth == max_depth
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))

        # Left and right subtrees for decision nodes
        if depth < max_depth:
            self.left = DDT(num_input, num_output, depth + 1, max_depth)
            self.right = DDT(num_input, num_output, depth + 1, max_depth)

    # Forward pass through the decision tree
    def forward(self, x, path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path  # Return policy at leaf node

        # Compute decision value for the node
        val = torch.sigmoid(torch.matmul(x, self.weights) + self.bias)

        # Recursive call to left and right subtrees
        left_output, left_path = self.left(x, path + "L")
        right_output, right_path = self.right(x, path + "R")

        # Combine outputs and paths based on decision value
        output = val * right_output + (1 - val) * left_output
        final_path = right_path if val >= 0.5 else left_path
        
        return output, final_path

# SubDDT (used by CoreScheduler) is a simplified version of the DDT
class SubDDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth):
        super(SubDDT, self).__init__()
        self.depth = depth
        self.max_depth = max_depth

        # Decision node
        if depth != max_depth:
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))

        # Leaf node
        if depth == max_depth:
            self.prob_dist = nn.Parameter(torch.zeros(num_output))

        # Left and right subtrees
        if depth < max_depth:
            self.left = SubDDT(num_input, num_output, depth + 1, max_depth)
            self.right = SubDDT(num_input, num_output, depth + 1, max_depth)

    # Forward pass through SubDDT
    def forward(self, x):
        if self.depth == self.max_depth:
            return self.prob_dist  # Return policy at leaf node

        # Compute decision value
        val = torch.sigmoid(torch.matmul(x, self.weights) + self.bias)

        # Recursive call to left or right subtree based on the decision value
        if val >= 0.5:
            return val * self.right(x)
        else:
            return (1 - val) * self.left(x)


import torch
from torch.optim import Adam

class SharedAdam(Adam):
    """
    Adam optimizer with shared states for multiprocessing environments.
    This allows parameters like step, exponential moving averages, and squared averages
    to be shared across multiple processes.
    """

    def __init__(self, params, lr=0.005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False, **kwargs):
        """
        Initializes the SharedAdam optimizer with shared states.
        
        Args:
        - params: Parameters to optimize.
        - lr: Learning rate for the optimizer (default: 0.01).
        - betas: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        - eps: Term added to denominator to improve numerical stability (default: 1e-8).
        - weight_decay: Weight decay (L2 penalty) (default: 0.0).
        - amsgrad: Whether to use the AMSGrad variant of this algorithm (default: False).
        - kwargs: Other keyword arguments to pass to the Adam optimizer.
        """
        # Initialize with standard Adam optimizer, passing all arguments
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)
        
        # Ensure that optimizer states are shared across processes
        self._share_memory()

    def _share_memory(self):
        """
        Moves the optimizer state to shared memory.
        This includes the step count, exp_avg (moving average of gradients),
        and exp_avg_sq (moving average of squared gradients).
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]

                    # Initialize shared state (move to shared memory)
                    state['step'] = torch.tensor(0.0).share_memory_()  # Shared step counter
                    state['exp_avg'] = torch.zeros_like(p.data).share_memory_()  # Shared exponential moving average of gradient
                    state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()  # Shared squared exponential average of gradient

    def share_memory(self):
        """
        Public method to move the state to shared memory, if needed.
        Can be called externally to ensure all optimizer states are shared.
        """
        self._share_memory()

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
        - closure: A closure that re-evaluates the model and returns the loss (optional).
        """
        # Iterate over parameter groups and apply optimization step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue  # Skip parameters with no gradients

                # Retrieve the shared state for this parameter
                state = self.state[p]
                
                # Increment the shared step count
                state['step'] += 1  
        
        # Call the original Adam step to perform weight updates
        super(SharedAdam, self).step(closure)




import torch
import torch.nn as nn

from model.trees.ClusTree import ClusTree
from model.trees.DDT import DDT
from model.trees.SoftDDT import SoftDDT

from configs import learning_config


def get_tree(devices):
    tree = learning_config['tree']
    max_depth = learning_config['tree_max_depth']
    if tree == "ddt":
        return DDT(num_input=get_num_input(), num_output=len(devices), depth=0, max_depth=max_depth)
    elif tree == "soft-ddt":
        return SoftDDT(num_input=get_num_input(), num_output=len(devices), depth=0, max_depth=max_depth)
    elif tree == "clustree":
        return ClusTree(num_input=get_num_input(), devices=devices, depth=0, max_depth=max_depth)


def get_num_input():
    num_input = 8
    if learning_config['onehot_kind']:
        num_input = 11
    if learning_config['utilization']:
        num_input +=2
    return num_input


def get_critic():
    num_input = get_num_input()

    if learning_config["learning_algorithm"] == "ppo" or learning_config["learning_algorithm"] == "a2c":
        num_hidden_layers = learning_config['critic_hidden_layer_num']
        critic_hidden_layer_dim = learning_config['critic_hidden_layer_dim']
        # Create list of layers
        layers = [nn.Linear(num_input, critic_hidden_layer_dim), nn.Sigmoid()]

        # Append hidden layers dynamically
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(critic_hidden_layer_dim, critic_hidden_layer_dim))
            layers.append(nn.Sigmoid())

        # Final output layer
        layers.append(nn.Linear(critic_hidden_layer_dim, 1))

        return nn.Sequential(*layers)
    else:
        return None
