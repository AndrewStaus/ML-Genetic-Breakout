'''
<h1>Train.py</h1>
<p>
This is the main training script.  It will train a new batch of agents.
<br>Results will be printed to the console and logged.

<ul>
<li>Population of 256 agents will be run through a genetic algorithm to create the next generation</li>
<li>Top Agent is saved every generation, Best Agent is saved each time a new highscore is achieved</li>
<li>Training will continue until the user quits</li>
</ul>
</p>
'''
from pandas.core.frame import DataFrame
from lib.neural_network import Tools
from concurrent.futures import ProcessPoolExecutor
from time import time
import lib.game_main as br
import pickle
import pandas as pd


def worker(agent, graphics = False) -> int:
    return br.main(agent=agent, graphics=graphics)


def process_generation(population: list) -> list:
    with ProcessPoolExecutor() as executor:
        fitness = list(executor.map(worker, population))

    return fitness, population

def log(training_log, generation, g_top_fit, top_fit, mean_fit, complete_time) -> DataFrame:
    training_log = training_log.append(pd.DataFrame({'Generation':[generation],
                                            'Top Fitness': [g_top_fit],
                                            'Mean Fitness': [mean_fit],
                                            'Run Time (s)': [complete_time]}))

 
    try:
        training_log.to_csv('./logs/training_log.csv', index = False)

    except Exception:
        pass

    print(f'Generation: {generation:03} -- Generation Top Fitness: {g_top_fit} -- All Time Top Fitness: {top_fit} --  Mean Fitness: {mean_fit:.2F} -- Completed in {complete_time:.2f} Seconds')
    return training_log


def main() -> None:
#######  Hyperparameters #######
    INPUT_SIZE = 25
    LAYER_CONFIG = [[16,'relu'],
                    [3,'softmax']]

    MUTATE_RATE = 0.25
    MUTATE_SCALE = 0.1

    POPULATION_SIZE = 256
################################


    generation = 1
    all_time_top_fitness = 0
    start_time = time()


    population = Tools.Genetic.create_population(POPULATION_SIZE, INPUT_SIZE, LAYER_CONFIG)

    training_log = pd.DataFrame(columns=['Generation', 'Top Fitness', 'Mean Fitness','Run Time (s)'])

    while True:

        fitnesses, population = process_generation(population)
        
        generation_top_fitness = max(fitnesses)
        indx_top_fitness = fitnesses.index(generation_top_fitness)

        #save best networks for replay
        #top agent is the the agent that had the highest score in the generation
        with open('./trained_agents/top_agent.pickle', 'wb') as f:
            pickle.dump(population[indx_top_fitness], f)
        if generation_top_fitness >= all_time_top_fitness:
            #best agent is the agent that has the highest score of the training session
            with open('./trained_agents/best_agent.pickle', 'wb') as f:
                pickle.dump(population[indx_top_fitness], f)
            all_time_top_fitness = generation_top_fitness

        #log to csv
        training_log = log(training_log, generation, generation_top_fitness, all_time_top_fitness, sum(fitnesses)/len(fitnesses), time()-start_time)
        start_time = time()       

        #Select best networks and perform crossover, mixing their weights and biases
        fitnesses = [fitness**2 for fitness in fitnesses]
        population = [Tools.Genetic.crossover(population, fitnesses) for _ in range(POPULATION_SIZE)]
        #mutate resulting children
        Tools.Genetic.mutate(population, rate = MUTATE_RATE, scale = MUTATE_SCALE)

        generation += 1



if __name__ == '__main__':
    main()
