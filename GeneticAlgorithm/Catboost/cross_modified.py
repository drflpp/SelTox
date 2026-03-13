import numpy as np
import random
import ga_compd_generation


in1 = [np.float64(29.4), 'chem_synthesis_reduction_by_D_maltose', np.float64(5.0), np.float64(27.8), 'MIC', np.float64(85.0), np.float64(30.0), 'DPPH', np.float64(24.0), 'tubular', np.float64(1.0), np.float64(11.0), 'soil', np.float64(1.7837651700316897), np.float64(107.868), np.float64(6.0), np.float64(1.0), np.float64(18.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), 'Mycobacterium smegmatis', np.float64(2.693861235821589), 'Mycobacterium smegmatis', np.float64(2.667660344843581), np.float64(0.02620089097800804), np.float64(14.788668355024598), np.float64(14.406224135714147)]
in2 =[np.float64(90.0), 'Green synthesis using Gloeophyllum striatum', np.float64(48.0), np.float64(50.1), 'MBEC', np.float64(65.0), np.float64(120.0), 'water (demineralized )', np.float64(30.0), 'oval', np.float64(1.0), np.float64(11.0), 'soil', np.float64(1.7837651700316897), np.float64(107.868), np.float64(6.0), np.float64(1.0), np.float64(18.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), 'Enterococcus hirae', np.float64(2.9913258709691153), 'Enterococcus hirae', np.float64(2.9891031029491684), np.float64(0.002222768019946919), np.float64(19.912065827265756), np.float64(19.867855077456902)]
indv2_list = ga_compd_generation.fitness(ga_compd_generation.population(size=50))
print(indv2_list.columns)
print(indv2_list.loc[2].values.tolist())
cross_over_frequency = 0.2
mutation_rate = 0.2




in1 = [np.float64(3.8), np.float64(7.913310681656804), np.float64(16.62106318045597), 'lab', 'MIC', 'small-molecule', 24, 'ethanol', 'spherical', 'liposome', 'polysaccharide', np.float64(1.243163121016122), 'weak', 'bio', np.float64(0.0), np.float64(11.0), np.float64(0.0), np.float64(0.0), 'meat', np.float64(0.0), np.float64(-0.2376), np.float64(48.0), np.float64(0.3194805194805194), np.float64(79.865), np.float64(60.0), np.float64(1.0), np.float64(1.7402597402597404), np.float64(1.0), np.float64(12.0), np.float64(1.0), np.float64(1.0), np.float64(0.0), 'clinical samples (blood, stool, wound swabs)', 'pathogenic', np.float64(0.0), 'nature', np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(0.0), np.float64(1.0), np.float64(0.0), np.float64(1.0), np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(1.0), 'Acinetobacter baumannii MTCC 1425', 'Green synthesis using Murraya koenigii']
in2 = [np.float64(1.4), np.float64(8.101809671168827), np.float64(14.149344825026265), 'commercial', 'MIC', 'none', 24, 'methanol', 'spherical', 'liposome', 'mixed', np.float64(4.688846486529344), 'medium', 'bio', np.float64(1.0), np.float64(17.0), np.float64(0.0), np.float64(1.0), 'clinical samples (sputum)', np.float64(0.0), np.float64(-1.118), np.float64(6.0), np.float64(0.7402597402597404), np.float64(196.967), np.float64(21.0), np.float64(1.0), np.float64(1.5194805194805197), np.float64(1.0), np.float64(14.0), np.float64(0.0), np.float64(1.0), np.float64(2.0), 'clinical samples (blood, stool, urine, wound swabs)', 'non-pathogenic', np.float64(1.0), 'human', np.float64(0.0), np.float64(0.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(1.0), np.float64(0.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(0.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(0.0), np.float64(0.0), np.float64(1.0), np.float64(0.0), np.float64(1.0), np.float64(1.0), 'Mycobacterium smegmatis nan', 'chemical_synthesis by aqueous zinc sulfate heptahydrate']

import random
material_list = ['amw', 'Source_origin', 'Capping_type', 'chi0v',  'Template_type',
 'Bio_component_class',  'Red_env_strength',
 'Red_env_type',  'Solvent_for_extract', 'Valance_electron',
 'Solvent_polar',  'lipinskiHBA',  'NumHBA',
 'CrippenClogP', 'hallKierAlpha', 'kappa1'
 ]


def to_crossover(indv1, indv2, cross_over_frequency):


    Size_group = [0, 1, 2] # np_size_max, np_size_min, np_size_avg,
    NP_group = [ 11, 15, 19, 20, 22, 23, 26, 31]  #  amw, Valence_electron, chi0v,np_synthesis
    # Extract_group = [4, 5, 7]     # Temperature, Duration, Solvent
    NP_group = [3, 5, 9, 10, 11, 12, 13, 15, 16, 19, 20, 22, 23, 26, 31,59]

    all_group = set(list(range(3, 58, 1)))
    independent_indices = list(all_group - set(NP_group))

    child = indv1.copy()

    if random.random() < cross_over_frequency:
        for i in NP_group:
            if i < len(child):
                child[i] = indv2[i]

    if random.random() < cross_over_frequency:
        for i in Size_group:
            if i < len(child):
                child[i] = indv2[i]

    # === Кроссовер по независимым признакам ===
    for i in independent_indices:
        if i < len(child) and random.random() < cross_over_frequency:
            child[i] = indv2[i]

    return child



# Example:
new_indv = to_crossover(in1, in2, cross_over_frequency=0.2)
print(new_indv)

print('\n size group')
indices = [0, 1, 2]
print('child: ',[new_indv[i] for i in indices]),
print('in1: ',[in1[i] for i in indices]),
print('in2: ',[in2[i] for i in indices])
print('\n nps group')
indices = [ 11, 19, 20, 22, 23, 26, 31]
print('child: ',[new_indv[i] for i in indices]),
print('in1: ',[in1[i] for i in indices]),
print('in2: ',[in2[i] for i in indices])
print('\n independent')
indices = [3, 4, 5, 6, 7, 8, 9, 10, 32, 33, 24]
print('child: ',[new_indv[i] for i in indices]),
print('in1: ',[in1[i] for i in indices]),
print('in2: ',[in2[i] for i in indices])

def to_mutation(individual1, mutation_rate):
    individual2 = indv2_list.iloc[random.randrange(20)].values.tolist()
    # print(individual2)
    mut = to_crossover(individual1, individual2, mutation_rate)
    return mut
