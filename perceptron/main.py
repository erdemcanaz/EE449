import random
import numpy as np
import matplotlib.pyplot as plt
from data_generators import Person
from perceptron import PerceptronRosenblatt


#==================
# Parameters
#==================
NUMBER_OF_EPOCHS = 1000
NUMBER_OF_TRAIN_SAMPLES = 250
NUMBER_OF_VALIDATION_SAMPLES = 10000
LEARNING_RATE = 0.01
PLOT_SAMPLES = True

TOTAL_SAMPLES = NUMBER_OF_TRAIN_SAMPLES + NUMBER_OF_VALIDATION_SAMPLES

#==================
# Generate samples
#==================
persons:list[Person] = []
for i in range(TOTAL_SAMPLES):
    gender = random.choice(["female","male"])
    persons.append(Person(gender = gender))

formatted_samples:list[list] = []

for person in persons:
    weight = person.get_weight()
    height = person.get_height()
    gender = 1 if person.get_gender() == "female" else -1

    formatted_samples.append([weight, height, gender])

#==================
# Train on samples
#==================
perceptron = PerceptronRosenblatt(number_of_inputs=2)
perceptron.update_learning_rate(LEARNING_RATE)

for epoch in range(NUMBER_OF_EPOCHS):
    print(f"\nEpoch: {epoch}")
    for formatted_sample in formatted_samples[0:NUMBER_OF_TRAIN_SAMPLES]:
        u = np.array( [[formatted_sample[0]], [formatted_sample[1]], [1]] )
        y = formatted_sample[2]
        perceptron.train_for_single_sample( u, y)

    #Verbose part
    perceptron.print_hyperplane_function()
    
    number_of_successes = 0
    number_of_failure = 0
    for formatted_sample in formatted_samples[NUMBER_OF_TRAIN_SAMPLES+1:]:
        u = np.array( [[formatted_sample[0]], [formatted_sample[1]], [1]] )
        y = formatted_sample[2]
        x = perceptron.calculate_output(u)
        if y != x:
            number_of_failure += 1
        else:
            number_of_successes += 1

    print(f"Number of successes: {number_of_successes}")
    print(f"Number of failures: {number_of_failure}")
    print(f"Success rate: {number_of_successes/(number_of_successes+number_of_failure)}")


#==================
# Validate result
#==================
if PLOT_SAMPLES:
    # Separating data based on gender
    male_data = [item for item in formatted_samples[0:NUMBER_OF_TRAIN_SAMPLES] if item[2] == -1]
    female_data = [item for item in formatted_samples[0:NUMBER_OF_TRAIN_SAMPLES] if item[2] == 1]

    # Unpacking the lists to separate height and weight
    male_heights, male_weights = zip(*[(item[1], item[0]) for item in male_data])
    female_heights, female_weights = zip(*[(item[1], item[0]) for item in female_data])

    # Plotting
    plt.scatter(male_weights, male_heights , color='blue', label='Male')
    plt.scatter(female_weights, female_heights, color='pink', label='Female')

    # Labeling axes
    plt.xlabel('Weight (kg)')
    plt.ylabel('Height (cm)')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()
    pass





