import argparse
import numpy as np
import torch

from energy_based_learning.datasets import load_dataloaders
from energy_based_learning.model.resistive.network import DeepResistiveEnergy
from energy_based_learning.model.function.network import Network
from energy_based_learning.model.resistive.minimizer import QuadraticMinimizer
from energy_based_learning.model.function.cost import SquaredError
from energy_based_learning.training.sgd import ContrastiveLearning
from energy_based_learning.training.epoch import Trainer, Evaluator
from energy_based_learning.training.monitor import Monitor, Optimizer



L = 10

graph = graph_utils.square_lattice(L, L)

