"""
A seq2seq model for
story training
"""

import math
import os
import random
import sys
import time
import random

from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf

import numpy as np

