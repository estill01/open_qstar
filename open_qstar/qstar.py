import sys
import torch
from torch import nn
from transformers import AutoModel
import networkx as nx
from loguru import logger
from open_qstar.graph_manager import GraphManager

logger.remove()
logger.add(sys.stderr, level="INFO")

class CoreTransformerModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.base_model(input_ids, attention_mask=attention_mask)

class DynamicCostQLearning(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.q_table = torch.zeros(state_dim, action_dim)
        self.state_graph = nx.Graph()
        self.graph_manager = GraphManager()

    def update_policy_and_graph(self, state, action, reward, next_state):
        # Update Q-table
        best_next_action = torch.argmax(self.q_table[next_state])
        td_target = reward + self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        # Update the graph
        self.update_graph(state, action, next_state, reward)
        self.graph_manager.add_or_update_edge(state, next_state, reward)

    def update_graph(self, state, action, next_state, reward):
        if not self.state_graph.has_node(state):
            self.state_graph.add_node(state)
        if not self.state_graph.has_node(next_state):
            self.state_graph.add_node(next_state)
        edge_weight = -reward
        self.state_graph.add_edge(state, next_state, weight=edge_weight)

def a_star_heuristic(state, goal_state):
    return torch.norm(state - goal_state)

class InstructionEncoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.encoder = nn.Linear(state_dim, state_dim)

    def forward(self, instruction_state):
        return self.encoder(instruction_state)

def update_goal_state_based_on_instruction(goal_state, instruction):
    return goal_state + instruction

class QStarModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', state_dim=1024, action_dim=30522):
        super().__init__()
        self.transformer_model = CoreTransformerModel(model_name)
        self.q_learning_module = DynamicCostQLearning(state_dim, action_dim)
        self.instruction_encoder = InstructionEncoder(state_dim)

    def forward(self, input_ids, attention_mask, instruction_ids, goal_state, reward=None, next_state=None):
        transformer_outputs = self.transformer_model(input_ids, attention_mask)
        current_state = transformer_outputs.last_hidden_state.mean(dim=1)

        instruction_state = self.transformer_model(instruction_ids, attention_mask=None).last_hidden_state.mean(dim=1)
        encoded_instruction = self.instruction_encoder(instruction_state)

        dynamic_goal_state = update_goal_state_based_on_instruction(goal_state, encoded_instruction)
        heuristic_value = a_star_heuristic(current_state, dynamic_goal_state)
        action = self.decide_action_with_a_star(current_state, dynamic_goal_state)

        if reward is not None and next_state is not None:
            self.q_learning_module.update_policy_and_graph(current_state, action, reward, next_state)

        return action

    def decide_action_with_a_star(self, current_state, goal_state):
        path = self.q_learning_module.graph_manager.shortest_path(current_state, goal_state)
        next_action = path[1] if len(path) > 1 else None
        return next_action

def process_stream_data(model, data_stream, optimizer):
    for data in data_stream:
        input_ids, attention_mask, instruction_ids, goal_state, reward, next_state = data
        action = model(input_ids, attention_mask, instruction_ids, goal_state, reward, next_state)
        # Define loss computation and update model
        # loss = ...
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()