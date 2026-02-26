import V4_cross_modified
import V4_ga_compd_generation
import random
import pandas as pd

population_size = 100
mutation_rate = 0.2
cross_over_rate = 0.2
df = V4_ga_compd_generation.fitness(V4_ga_compd_generation.population(population_size)).sort_values('Fitness', ascending=False)
df = df.reset_index(drop=True)

def evolve_crossing(df_compound_list, cross_over_rate, mutation_rate):
    original = df_compound_list
    unique = []
    length = len(original)-1
    j = 0
    while j < length:
        if str(original.iloc[[j], [0]].values) == str(original.iloc[[j+1],[0]].values): #keeping only unique np
            unique.append(original.iloc[j].values.tolist())
            j += 1
        else:
            unique.append(original.iloc[j].values.tolist())
        j+= 1
    dff = pd.DataFrame(unique, columns=original.columns)

    #crossing two individuals
    i = 0
    selected_ind = []
    while i < len(dff):
        individual1 = dff.iloc[i].values.tolist()
        individual2 = dff.iloc[random.randint(0, len(dff) -1)].values.tolist()
        # individual1 = unique[i]
        # individual2 = unique[random.randrange(0, len(unique), 1)]
        # print(individual1,'\n',individual2)
        #cross_individual = cross_normal.crossover_individuals(individual1, individual2, cross_over_rate)
        cross_individual = V4_cross_modified.to_crossover(individual1, individual2, cross_over_rate)
        mutate_individual = V4_cross_modified.to_mutation(cross_individual, mutation_rate)
        selected_ind.append(mutate_individual)
        i +=1
    dframe =pd.DataFrame(selected_ind, columns=df_compound_list.columns)
    #dframe is crossed (new gen) population
    dframe_copy = dframe.copy()
    dframe_copy= dframe_copy.iloc[: , :-6]
    dframe_evolved = V4_ga_compd_generation.fitness(dframe_copy)
    #check fitness of new generation
    selection = []
    for a in range(len(dff)):
        if dff.iloc[a,-1] >= dframe_evolved.iloc[a,-3]:
            selection.append(dff.iloc[a].values.tolist())
        else:
            selection.append(dframe_evolved.iloc[a].values.tolist())
    #select parents if they have high fitness (drop children) or select children if parents have low fitness
    select_single = []

    df_new = pd.DataFrame(selection, columns=df_compound_list.columns)
    all_sort = df_new.sort_values('Fitness', ascending=False)
    all_sort.reset_index(drop=True, inplace=True)
    without_selection = pd.DataFrame(dframe_evolved, columns=df_compound_list.columns)
    # print(original, all_sort)
    return all_sort

# print('here',evolve_crossing(df, cross_over_rate, mutation_rate), evolve_crossing(df, cross_over_rate, mutation_rate).columns)