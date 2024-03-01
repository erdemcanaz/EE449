import math
import numpy as np

class Person():
        
    def __init__(self, gender = None):
        if gender not in ["male", "female"]:
            raise Exception("Gender must be either male or female")        
        self.GENDER = gender

        #BMI = weight (kg) / (height (m))^2
        self.HEIGHT_m = round(self.__sample_gaussian_height_distribution(gender = self.GENDER)/100,2)
        self.BMI = round(self.__sample_gaussian_BMI_distribution(gender = self.GENDER),2)
        self.WEIGHT_kg = round(self.BMI * self.HEIGHT_m**2,2)

    def __sample_gaussian_height_distribution(self, gender:str="female"):
        """
        Sample a height from a Gaussian distribution with a given mean and standard deviation.
        #Approximated according to: https://ourworldindata.org/human-height

        Parameters:
        - mu: The mean of the Gaussian distribution (164.7 and 178.4 for famele and male respectively)
        - sigma: The standard deviation of the Gaussian distribution (7.1 and 7.6 for female and male respectively).

        Returns:
        - A height sampled from the corresponding Gaussian distribution.
        """

        mu = 150 if gender=="female" else 178.4
        sigma = 5 if gender=="female" else 5

        # Sample a single value from the Gaussian distribution
        return np.random.normal(mu, sigma)

    def __sample_gaussian_BMI_distribution(self, gender:str="female"):
        """
        Sample a BMI from a Gaussian distribution with a given mean and standard deviation.
        #Approximated according to: https://www.fao.org/3/T1970E/t1970e08.htm

        Parameters:
        - mu: The mean of the Gaussian distribution (24.30 and 24.71 for famele and male respectively)
        - sigma: The standard deviation of the Gaussian distribution (3.86 and 4.40 for female and male respectively).

        Returns:
        - A height sampled from the corresponding Gaussian distribution.
        """

        mu = 24.30 if gender=="female" else 24.71
        sigma = 3.86 if gender=="female" else 4.40

        # Sample a single value from the Gaussian distribution
        return np.random.normal(mu, sigma)
    
    def get_height(self):
        return self.HEIGHT_m
    
    def get_weight(self):
        return self.WEIGHT_kg
    
    def get_gender(self):
        return self.GENDER
    


        