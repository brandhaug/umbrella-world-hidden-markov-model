import numpy as np


def backward(backward_message, evidence_value):
    new_backward_message = 0


    return new_backward_message


def normalize(param):
    smoothed_estimate = 0

    return smoothed_estimate


def forward(forward_message, evidence_value):
    new_forward_message = 0

    return new_forward_message


# Smoothing: computes posterior probabilities of a sequence of states given a sequence of observation
# returns a vector of probability distributions
def forward_backward(evidence_values, prior_distribution):
    forward_messages[0] = prior_distribution
    for i in range(1, time_slice):
        forward_messages[i] = forward(forward_messages[i - 1], evidence_values[i])

    for i in range(time_slice, 1, -1):
        smoothed_estimates[i] = normalize(forward_messages[i] * backward_message)
        backward_message = backward(backward_message, evidence_values[i])

    return probability_distribution


dynamic_model = np.array([
    [0.7, 0.3],
    [0.3, 0.7]
])

transition_model = np.array([
    [0.9, 0.3],
    [0.3, 0.9]
])

evidence_values = [True, True]  # a vector of evidence values for steps 1, ..., t
prior_probability_distribution = []  # the prior distribution on the initial state, P(X0)
probability_distribution = []
smoothed_estimates = []  # a vector of smoothed estimates for steps 1, ..., t
backward_message = 0  # a representation of the backward message, initially all 1s
forward_messages = []  # a vector of forward messages for steps 0, ..., t
time_slice = 0

if __name__ == '__main__':
    probability_distribution = forward_backward(evidence_values, prior_probability_distribution)
