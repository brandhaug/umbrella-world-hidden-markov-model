import numpy as np


def forward(observations):
    initial_probabilities = np.array([0.5, 0.5])
    forward_list = [initial_probabilities]
    rain_t_minus_1_given_observation_t_minus_1 = initial_probabilities

    # Looping through days
    index = 1
    for observation in enumerate(observations):

        # Calculating P(Xt | Et-1) by dotting P(Xt | Xt-1) with (Xt-1 | Et-1)
        rain_t_given_umbrella_minus_1 = np.dot(transition_probabilities, rain_t_minus_1_given_observation_t_minus_1)
        # print("Rain(X{}|e{}) = {}".format(index, (index - 1), rain_t_given_umbrella_minus_1))

        # If umbrella is observed, dot P(Et | Xt) with P(Xt | Et-1)
        if observation[1] == possible_observations['Umbrella']:
            rain_t_given_observation_t = np.dot(emission_probabilities_umbrella, rain_t_given_umbrella_minus_1)
        # If umbrella is not observed, dot P(~Et | Xt) with P(Xt | Et-1)
        else:
            rain_t_given_observation_t = np.dot(emission_probabilities_no_umbrella, rain_t_given_umbrella_minus_1)

        # Normalizing result, which is P(Xt | Et)
        rain_t_given_observation_t = rain_t_given_observation_t / rain_t_given_observation_t.sum()
        print("P(X{}|e1:{}) = {}".format(index, index, rain_t_given_observation_t))

        # Add to forward list for further processing in smoothing
        forward_list.append(rain_t_given_observation_t)

        # Set P(Xt-1 | Et-1) to P(Xt | Et). In other words, set current yesterday to today before next iteration
        rain_t_minus_1_given_observation_t_minus_1 = rain_t_given_observation_t

        index += 1

    return forward_list


def backward(observations, print_output):
    # Reversing observations
    observations = observations[::-1]

    # Initial value
    b_hat = np.array([1.0, 1.0])

    if print_output:
        print("Backwards {}:{} = {}".format(len(observations), len(observations), b_hat))

    # Set list of all backward calculations - P(Ek+1:t | Xk)
    backwards_list = [b_hat]

    # Looping through days
    for i, observation in enumerate(observations):

        # If umbrella is observed, dot P(Et | Xt) with P(Ek+1:t | Xk)
        if observation == possible_observations['Umbrella']:
            a = np.dot(emission_probabilities_umbrella, b_hat)
        # If umbrella is not observed, dot P(~Et | Xt) with P(Ek+1:t | Xk)
        else:
            a = np.dot(emission_probabilities_no_umbrella, b_hat)

        # Dotting a with P(Xt | Xt-1)
        b = np.dot(a, transition_probabilities)

        if print_output:
            print("Backwards {}:{} = {}".format(len(observations) - i - 1, len(observations), b))

        # Normalizing
        b = b / b.sum()

        # Adding to list for further processing in smoothing
        backwards_list.append(b)

        # Set last value to current value
        b_hat = b

    return backwards_list


def smoothing(forward_list, backward_list):
    # Reversing backwards list
    backward_list = backward_list[::-1]

    # Looping through backwards list
    for index, b in enumerate(backward_list):
        # Multiplying the element form the forward list with the element in the reversed backwards list
        rain_k_given_observation_e_1_t = np.multiply(forward_list[index], b)

        # Normalizing
        rain_k_given_observation_e_1_t = rain_k_given_observation_e_1_t / rain_k_given_observation_e_1_t.sum()

        if index > 0:
            print("P(X{}|e{}:{}) = {}".format(len(backward_list) - index, 1, len(forward_list) - 1,
                                              rain_k_given_observation_e_1_t))


possible_observations = {
    'Umbrella': True,
    'No Umbrella': False
}

transition_probabilities = np.array([[0.7, 0.3],
                                     [0.3, 0.7]])

emission_probabilities_umbrella = np.array([[0.9, 0],
                                            [0, 0.2]])

emission_probabilities_no_umbrella = np.array([[0.1, 0],
                                               [0, 0.8]])

observations_1 = [True, True]
observations_2 = [True, True, False, True, True]

if __name__ == '__main__':
    print("Forward probabilities (Task B1):")
    forward_1 = forward(observations_1)
    print("\nForward probabilities (Task B2):")
    forward_2 = forward(observations_2)

    backward_1 = backward(observations_1, False)
    print("\nSmoothing probabilities (Task C1):")
    smoothing(forward_1, backward_1)

    print("\nBackward probabilities (Task C2):")
    backward_2 = backward(observations_2, True)
    print("\nSmoothing probabilities (Task C2):")
    smoothing(forward_2, backward_2)
