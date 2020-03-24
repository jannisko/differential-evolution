import tensorflow as tf
import numpy as np

class DifferentialEvolution():

    def __init__(self,
        loss_function,
        optimization_variables,
        boundaries=10,
        count_population=20,
        differential_weight=0.8,
        crossover_probability=0.7
    ):


        self.loss_function = loss_function
        self.optimization_variables = optimization_variables
        self.boundaries = boundaries
        self.count_population = count_population
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability

        initial_population = [
            [
                tf.random.uniform(
                    shape=[1],
                    minval=-self.boundaries,
                    maxval=self.boundaries
                ) 
                for _ in range(len(self.optimization_variables))
            ] 
            for _ in range(self.count_population)
        ] 
        
        self.current_population = initial_population

    def _compute_loss(self, new_variables):

        for variable, new_value in zip(self.optimization_variables, new_variables):
            variable.assign(new_value[0])
    
        loss = self.loss_function()
        return loss

    def get_best_point(self):
        best_loss = np.inf
        best_point = None
        for point in self.current_population:
            loss = self._compute_loss(point)
            if(loss < best_loss):
                best_loss = loss
                best_point = point
        return best_point, best_loss

    def next_generation(self):

        for j in range(self.count_population):
            # choose 3 other points randomly
            p_old = self.current_population[j]
            indexes = [index for index in range(self.count_population) if index != j]
            rand_indexes = tf.random.shuffle(indexes)
            p1 = self.current_population[rand_indexes[0]]
            p2 = self.current_population[rand_indexes[1]]
            p3 = self.current_population[rand_indexes[2]]
            p_new = [
                tf.clip_by_value(
                    p1[i] + self.differential_weight * (p2[i] - p3[i]),
                    -self.boundaries,
                    self.boundaries
                )
                for i in range(len(self.optimization_variables))
            ]

            replacement_indexes = tf.random.uniform(shape=[len(self.optimization_variables)]) < self.crossover_probability

            p_mixed = [
                p_old[i] if replacement_bool else p_new[i]
                for i, replacement_bool in enumerate(replacement_indexes)
            ]
            loss_mixed = self._compute_loss(p_mixed)
            loss_old = self._compute_loss(p_old)

            if(loss_mixed < loss_old):
                self.current_population[j] = p_mixed
