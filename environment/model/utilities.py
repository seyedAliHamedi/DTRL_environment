
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


class SharedAdam(torch.optim.Adam):
    """
    Adam optimizer with shared states for multiprocessing environments.
    This allows parameters like step, exponential moving averages, and squared averages
    to be shared across multiple processes.
    """

    def __init__(self, params, lr=0.005):
        """
        Initializes the SharedAdam optimizer.
        
        Args:
        - params: Parameters to optimize.
        - lr: Learning rate for the optimizer.
        """
        # Initialize with standard Adam optimizer
        super(SharedAdam, self).__init__(params, lr=lr)
        
        # Share memory across processes for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Initialize shared state
                    state = self.state[p]
                    state['step'] = torch.tensor(0.0).share_memory_()  # Shared step counter
                    state['exp_avg'] = torch.zeros_like(p.data).share_memory_()  # Shared exponential moving average
                    state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()  # Shared squared exponential average

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
