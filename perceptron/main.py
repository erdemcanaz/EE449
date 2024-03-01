import random
import numpy as np
import matplotlib.pyplot as plt
from data_generators import Person
from perceptron import PerceptronRosenblatt


#==================
# Parameters
#==================
NUMBER_OF_EPOCHS_BEFORE_REDUCING_SUCCES_CRITERIA = 50
NUMBER_OF_TRAIN_SAMPLES = 1000
NUMBER_OF_VALIDATION_SAMPLES = 1000
LEARNING_RATE = 0.001
SUCCES_CRITERIA = 1.0
SUCCES_CRITERIA_STEP = 0.01
PLOT_SAMPLES = True

WEIGHT_NORMALIZER = 80
HEIGHT_NORMALIZER = 1.8

TOTAL_SAMPLES = NUMBER_OF_TRAIN_SAMPLES + NUMBER_OF_VALIDATION_SAMPLES

#==================
# Generate samples
#==================

MIN_WEIGHT = None
MAX_WEIGHT = None
persons:list[Person] = []
for i in range(TOTAL_SAMPLES):
    gender = random.choice(["female","male"])
    persons.append(Person(gender = gender))

formatted_samples:list[list] = []

for person in persons:
    weight = person.get_weight()
    normalized_weight = weight / WEIGHT_NORMALIZER
    height = person.get_height()
    normalized_height = height / HEIGHT_NORMALIZER
    gender = 1 if person.get_gender() == "female" else -1

    if MIN_WEIGHT == None or weight < MIN_WEIGHT:
        MIN_WEIGHT = weight
    if MAX_WEIGHT == None or weight > MAX_WEIGHT:
        MAX_WEIGHT = weight
        
    formatted_samples.append([normalized_weight, normalized_height, gender])

#==================
# Train on samples
#==================
perceptron = PerceptronRosenblatt(number_of_inputs=2)
perceptron.update_learning_rate(LEARNING_RATE)

while SUCCES_CRITERIA>0:
    is_criteria_satisfied = False
    for epoch in range(NUMBER_OF_EPOCHS_BEFORE_REDUCING_SUCCES_CRITERIA):
        print(f"\nEpoch: {epoch} Succes Criteria: {SUCCES_CRITERIA}")
        for formatted_sample in formatted_samples[0:NUMBER_OF_TRAIN_SAMPLES]:
            u = np.array( [[formatted_sample[0]], [formatted_sample[1]], [1]] )
            y = formatted_sample[2]
            perceptron.train_for_single_sample( u, y)

        #Verbose part
        perceptron.print_hyperplane_function()
        
        number_of_successes = 0
        number_of_failure = 0
        for formatted_sample in formatted_samples[NUMBER_OF_TRAIN_SAMPLES:]:
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

        if number_of_successes/(number_of_successes+number_of_failure) > SUCCES_CRITERIA:
            print(f"Success rate reached the success criteria of {SUCCES_CRITERIA}")
            is_criteria_satisfied = True
            break
    
    SUCCES_CRITERIA -= SUCCES_CRITERIA_STEP
    if is_criteria_satisfied:
        break
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

    x = np.linspace(MIN_WEIGHT/WEIGHT_NORMALIZER, MAX_WEIGHT/WEIGHT_NORMALIZER, 100)
    y = []
    for item in x:
        y.append(perceptron.seperation_line_function_for_2_input(item))
  
    plt.plot(x, y, color='black', label='Perceptron')
    # Labeling axes
    plt.xlabel('Weight (kg if not normalized)')
    plt.ylabel('Height (cm if not normalized)')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()
    pass





