import pandas as pd
import V4_ga_compd_generation
import V4_crossing_mutation
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from V4_ga_compd_generation import name_of_good_bacteria, name_of_pathogenic_bacteria

# folder_name = 'P.aeruginosa_vs_B.subtilis'
folder_name=f'{name_of_pathogenic_bacteria}_vs_{name_of_good_bacteria}'
# The path where you want to save your file
folder_path = f"output/{folder_name}"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)


mutation_rate = 0.2
cross_over_rate = 0.2


#import warnings
#warnings.filterwarnings("ignore")
# df = ga_compd_generation_new.fitness(ga_compd_generation_new.population(population_size)).sort_values('Fitness', ascending=False)

def new_generations(Gen, population_size):
    half = int((population_size * 0.5)+1)
    selected = Gen.iloc[:half,:]
    new = [selected, V4_ga_compd_generation.fitness(V4_ga_compd_generation.population(half))]
    new_generation_input = pd.concat(new)
    new_generation_input.reset_index(drop=True, inplace=True)
    new_gen = V4_crossing_mutation.evolve_crossing(new_generation_input, cross_over_rate, mutation_rate)
    new_gen.reset_index(drop=True, inplace=True)
    return new_gen


# print('original', df, 'new', new_generations(df, population_size))
means = []
maxs = []
def Genetic_Algorithm(generation_number, population_size):
    Generation1 = V4_ga_compd_generation.fitness(V4_ga_compd_generation.population(population_size)).sort_values('Fitness', ascending=False)
    mean1 = Generation1['Fitness'].mean()
    max1 = Generation1['Fitness'].max()
    Generation1.to_csv(f'output/{folder_name}/pop_size_' + str(population_size) + '_Generation_1.csv')
    Generation2 = V4_crossing_mutation.evolve_crossing(Generation1, cross_over_rate, mutation_rate)
    mean2 = Generation2['Fitness'].mean()
    max2 = Generation2['Fitness'].max()
    Generation2.to_csv(f'output/{folder_name}/pop_size_' + str(population_size)+ '_Generation_2.csv')
    Generation_next = Generation2
    means = [ mean1, mean2]
    maxs = [max1, max2]
    g = 3
    while g in range(generation_number + 1):
        Generation_next = new_generations(Generation_next, population_size)
        # i = Generation_next.iloc[0][0]
        mean = Generation_next['Fitness'].mean()
        max = Generation_next['Fitness'].max()
        Generation_next.to_csv(f'output/{folder_name}/pop_size_' + str(population_size) + '_Generation_' + str(g) + '.csv')
        means.append(mean)
        maxs.append(max)

        g += 1

    genn = generation_number + 1
    gens = list(range(1,genn))
    summary = pd.DataFrame( list(zip( gens, means, maxs)), columns= ['generations','mean', 'max'] )
    print(summary)
    #summary.to_csv('output/results/pop_size_50/t2/summary_pop_size_' + str(population_size) + '.csv')
    summary.to_csv(f'output/{folder_name}/summary_pop_size_' + str(population_size) +'_gen_' + str(generation_number)+'.csv')

    fig = plt.figure()
    x = summary['generations']
    y = summary['mean']
    z = summary['max']
    plt.plot(x, y, 'bo', label='mean')
    plt.plot(x, z, 'ro', label='max')
    plt.legend(loc='lower right')
    fig.savefig(f'output/{folder_name}/summary_pop_size_' + str(population_size) +'_gen_' + str(generation_number)+'.png')
    return Generation_next


def final_loop():
    pop_col = []
    time_all = []
    gen_col = []
    gen = 100
    while gen <= 100:
        population_size = 100
        while population_size <= 100:
            st = time.time()
            Genetic_Algorithm(gen, population_size)
                #population_size += 10
            gen_col.append(gen)
                # gen += 10
            escape_time = time.time() - st
            time_all.append(escape_time)
            pop_col.append(population_size)
            print('Escape time:', escape_time)
            population_size += 10
        gen +=10
        et = pd.DataFrame(list(zip(pop_col, gen_col, time_all)), columns=['population_size','Generation number', 'Time'])
        # et.to_csv('output/results/pop_size_50/t2/Time_' + str(population_size-10) + '.csv')
        et.to_csv(f'output/{folder_name}/pop_size_' + str(population_size) + '.csv')




final_loop()

file_path_1 = r"D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC_XGB\output\P.aeruginosa_vs_B.subtilis\pop_size_100_Generation_100.csv"
master_path_1 = r"D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC_XGB\output\P.aeruginosa_vs_B.subtilis\results_for_runs_Saur_vs_Bsubt.csv"


from pathlib import Path


def get_results_file(file_path, master_file_path):
    """
    Reads the first 31 rows from a run CSV and appends them to a unified master file.

    Parameters:
        file_path (str or Path): Path to the run CSV file.
        master_file_path (str or Path): Path to the unified master CSV file.
    """
    # Read the CSV
    df = pd.read_csv(file_path)

    # Take only the first 31 rows
    df = df.iloc[:31]

    # Ensure master file exists
    master_file = Path(master_file_path)
    if master_file.exists():
        # Append to existing master file
        df.to_csv(master_file, mode='a', index=False, header=False)
    else:
        # Create a new master file
        df.to_csv(master_file, index=False)

    print(f"Appended results from {file_path} to {master_file_path}")


# Example usage:
# file_path_1 = r"D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\output\P.aeruginosa_vs_B.subtilis\pop_size_100_Generation_100.csv"
# master_file = r"D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\output\master_results.csv"

get_results_file(file_path_1, master_path_1)

# print(pd.read_csv(master_path_1))
# file1 = "D:\NPs_Platform_df1\NPs_Platform_df1\V4_MIC\output\P.aeruginosa_vs_B.subtilis\pop_size_100_Generation_100.csv"