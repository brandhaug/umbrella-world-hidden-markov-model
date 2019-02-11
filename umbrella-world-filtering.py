import numpy as np


# def backward(backward_message, evidence_value):
#     new_backward_message = 0
#
#     return new_backward_message
#
#
# def normalize(param):
#     smoothed_estimate = 0
#
#     return smoothed_estimate
#
# # Smoothing: computes posterior probabilities of a sequence of states given a sequence of observation
# # returns a vector of probability distributions
# def forward_backward(evidence_values, prior_distribution):
#     forward_messages[0] = prior_distribution
#     for i in range(1, time_slice):
#         forward_messages[i] = forward(forward_messages[i - 1], evidence_values[i])
#
#     for i in range(time_slice, 1, -1):
#         smoothed_estimates[i] = normalize(forward_messages[i] * backward_message)
#         backward_message = backward(backward_message, evidence_values[i])
#
#     return probability_distributions


# Transform the probability distributions related to a given hidden Markov model into matrix notation

def forward(observations, states, start_prob, end_st):
    forward_message = []
    forward_message_current = {}
    forward_message_previous = {}
    for i, observation in enumerate(observations):
        forward_message_current = {}
        for state in states:
            if i == 0:
                forward_previous_sum = start_prob[state]
            else:
                forward_previous_sum = sum(forward_message_previous[k] * transition_probabilities[k][state] for k in states)

            forward_message_current[state] = emission_probabilities[state][observation] * forward_previous_sum

        forward_message.append(forward_message_current)
        forward_message_previous = forward_message_current

    return sum(forward_message_current[k] * transition_probabilities[k][end_st] for k in states)


# State matrix / (T): probability distributions
# Column (i) = Target state
# Row (j) = Start state
probability_distributions = np.array([
    [0.7, 0.3],
    [0.3, 0.7]
])

# Event matrix (B): probabilities for observing events given a particular state
evidence_values = np.array([
    [0.9, 0.3],
    [0.3, 0.9]
])

states = ('Rain', 'Not Rain')
observations = ('Umbrella', 'Not Umbrella')

transition_probabilities = {
    'Rain': {'Rain': 0.7, 'Not Rain': 0.3},
    'Not Rain': {'Not Rain': 0.7, 'Rain': 0.3},
}

emission_probabilities = {
    'Rain': {'Umbrella': 0.7, 'Not Umbrella': 0.3},
    'Not Rain': {'Not Umbrella': 0.7, 'Umbrella': 0.3},
}

start_probability = transition_probabilities['Rain']
emission_probability = emission_probabilities['Rain']

evidence_values = [True, True]  # a vector of evidence values for steps 1, ..., t
prior_probability_distribution = []  # the prior distribution on the initial state, P(X0)
probability_distributions = []
smoothed_estimates = []  # a vector of smoothed estimates for steps 1, ..., t
backward_message = 0  # a representation of the backward message, initially all 1s
forward_messages = []  # a vector of forward messages for steps 0, ..., t
time_slice = 0

if __name__ == '__main__':
    forward(observations, states, start_probability, transition_probabilities, emission_probabilities, 0)
