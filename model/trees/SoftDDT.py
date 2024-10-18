import torch
import torch.nn as nn
import torch.optim as optim
from configs import learning_config
from environment.util import extract_pe_data
class SoftDDT(nn.Module):
    def __init__(self, num_input, num_output, depth, max_depth):
        """
        Initializes the SoftDDT structure.

        Args:
            num_input (int): Number of input features.
            num_output (int): Number of output classes or probabilities.
            depth (int): Current depth of the tree.
            max_depth (int): Maximum depth of the tree.
        """
        super(SoftDDT, self).__init__()

        # Store depth-related properties
        self.depth = depth
        self.max_depth = max_depth
        
        if depth < max_depth:
            # Initialize weights and biases for internal nodes
            self.weights = nn.Parameter(torch.empty(num_input).normal_(mean=0, std=0.1))
            self.bias = nn.Parameter(torch.zeros(1))

            # Recursively create left and right child nodes
            self.left = SoftDDT(num_input, num_output, depth + 1, max_depth)
            self.right = SoftDDT(num_input, num_output, depth + 1, max_depth)
        else:
            # Leaf node stores output probabilities for the output classes
            self.prob_dist = nn.Parameter(torch.ones(num_output))
            self.logit_regressor = nn.Sequential(
                nn.Linear(learning_config['pe_num_features'],128),
                nn.Sigmoid(),
                nn.Linear(128,128),
                nn.Sigmoid(),
                nn.Linear(128,1),
            )
            self.logit_optimizer = optim.Adam(self.logit_regressor.parameters(),lr=0.01)

    def forward(self, x, path=""):
        if self.depth == self.max_depth:
            return self.prob_dist, path
        
        # Compute decision value using weights and bias
        val = torch.sigmoid((torch.matmul(x, self.weights) + self.bias))

        # Recursive calls to left and right subtrees
        left_output, left_path = self.left(x, path + "L")
        right_output, right_path = self.right(x, path + "R")

        # Combine outputs and paths based on decision value
        output = val * right_output + (1 - val) * left_output
        final_path = right_path if val >= 0.5 else left_path
        
        return output, final_path

            
    def add_device(self, new_device):
        if self.depth == self.max_depth:
            # Get the features of the new device
            new_device_features = extract_pe_data(new_device)

            # Ensure the features are in the right shape for the regressor (2D array)
            # Predict logit using the logit regressor
            new_device_dist = self.logit_regressor(torch.tensor(new_device_features,  dtype=torch.float32))

            # Logging predicted and average logits for debugging
            avg_logit = sum(self.prob_dist) / len(self.prob_dist)
            print(f"Avg Logit: {avg_logit:.4f}, Predicted Logit: {new_device_dist,}")

            
            # Concatenate the new distribution and wrap it in nn.Parameter
            self.prob_dist = nn.Parameter(torch.cat((self.prob_dist, new_device_dist)))

        else:
            # Recursively call add_device on the left and right subtrees
            self.left.add_device(new_device)
            self.right.add_device(new_device)


            
    def remove_device(self, device_index):
        if self.depth == self.max_depth:
            # Remove the device's entry in prob_dist
            self.prob_dist = nn.Parameter(torch.cat((self.prob_dist[:device_index], self.prob_dist[device_index+1:])))
            
            # Re-normalize the probabilities to ensure they sum to 1
            # with torch.no_grad():
            #     self.prob_dist /= torch.sum(self.prob_dist)
        else:
            self.left.remove_device(device_index)
            self.right.remove_device(device_index)

    