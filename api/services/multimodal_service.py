# This file is responsible for calculating the mode choice probabilities for the riders between three modes of transport:- bike, car, auto. The predict_mode_share function is solely responsible for calculation of the probabilities. This function takes the inputs of distance, time and surge multiplier. It then first calculates the cost of the travel, time of travel by three modes of transport with formulas mentioned. It also assigns some time based utility based adjustments as riders preffer to travel by car in the late nights, and by bike during morning rush hours. It then calculates the utilities, softmax probabilities based on those utilities and then calculates individual probabilities by dividing the individual exponentiated utilities by the sum of all exponentiated utilities. Thhen at last returns the dictionary of the cost, probability, and durations.

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class MultimodalService:
    def __init__(self):
        self.beta_cost = -0.22   
        self.beta_time = -0.08

        self.asc_car = 0.6       
        self.asc_auto = 0.0      
        self.asc_bike = -0.8    

    def predict_mode_share(
        self,
        distance_miles: float,
        travel_time_minutes: float,
        surge_multiplier: float,
        hour: int = 12,
        base_fare: float = 15.0,
    ) -> dict:
        
        distance_miles = max(0.1, distance_miles)
        travel_time_minutes = max(1.0, travel_time_minutes)

        #Calculation of travel costs
        car_cost = base_fare * surge_multiplier
        auto_cost = 3.00 + (3.50 * distance_miles)
        bike_cost = 4.00

        #Calculation of travel times
        car_time = travel_time_minutes
        auto_time = travel_time_minutes
        bike_time = (distance_miles * 6.0) + 3.0

        #Applying time-dependent utility adjustments
        asc_bike_adj = self.asc_bike
        asc_car_adj = self.asc_car

        #Nighttime penalty for biking
        if hour in [22, 23, 0, 1, 2, 3, 4, 5]:
            asc_bike_adj -= 1.5  # Biking is less attractive at night due to safety/darkness
            asc_car_adj += 0.3   

        #Rush hour adjustments
        elif hour in [7, 8, 9, 17, 18, 19]:
            asc_bike_adj += 0.4
            asc_car_adj -= 0.2

        #Calculation of utilities
        u_car = asc_car_adj + (self.beta_cost * car_cost) + (self.beta_time * car_time)
        u_auto = self.asc_auto + (self.beta_cost * auto_cost) + (self.beta_time * auto_time)
        u_bike = asc_bike_adj + (self.beta_cost * bike_cost) + (self.beta_time * bike_time)

        #Computing softmax probabilities
        exp_car = math.exp(u_car)
        exp_auto = math.exp(u_auto)
        exp_bike = math.exp(u_bike)
        sum_exp = exp_car + exp_auto + exp_bike

        prob_car = exp_car / sum_exp
        prob_auto = exp_auto / sum_exp
        prob_bike = exp_bike / sum_exp

        return {
            "probabilities": {
                "car": round(prob_car, 4),
                "auto": round(prob_auto, 4),
                "bike": round(prob_bike, 4),
            },
            "costs": {
                "car": round(car_cost, 2),
                "auto": round(auto_cost, 2),
                "bike": round(bike_cost, 2),
            },
            "durations": {
                "car": round(car_time, 1),
                "auto": round(auto_time, 1),
                "bike": round(bike_time, 1),
            },
        }

multimodal_service = MultimodalService()
