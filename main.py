##################################################################
#
# Andrew Baker
#
# File Name:  main.py
# Date:  01/06/17
#
# Description:  The main function of a simulated neural network.
#
#
##################################################################

from neural_flow import *

x, y, z = Input(), Input(), Input()
inputs = [x, y, z]

# f = Add(x, y, z)
weight_x, weight_y, weight_z = Input(), Input(), Input()
weights = [weight_x, weight_y, weight_z]

bias = Input()

f = Linear(inputs, weights, bias)

feed_dict = {x: 6,
             y: 14,
             z: 3,
             weight_x: 0.5,
             weight_y: 0.25,
             weight_z: 1.4,
             bias: 2}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# output should be 12.7
print(output)
