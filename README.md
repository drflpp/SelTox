**Abstract**

The rational design of nanoparticles (NPs) with selective antimicrobial activity remains a major challenge in nanomedicine, particularly for combating antimicrobial resistance while preserving beneficial microbiota. In this work, we present an AI-driven discovery platform that integrates predictive modeling and generative design to prioritize selectively antibacterial inorganic NPs, which are subsequently validated experimentally. The platform combines a curated dataset of 2098 antibacterial activity measurements with enriched descriptors, including physicochemical properties, synthesis parameters, and Kyoto Encyclopedia of Genes and Genomes (KEGG) derived functional orthologs and metabolic pathways. Gradient boosting models trained on this gene informed dataset achieved high predictive performance, especially XGB regression acheived cross-validated R² score of 0.79, 0.83 and 0.80 for predicting minimal concentration (MC) with experimental parameters, functional ortholog features and metabolic pathways. Next, GenSelTox couples these models with a generative framework, integrating genetic algorithms, and reinforcement learning to explore NP formulation space for optimal selective toxicity. As proof-of-concept, one of top-ranked ZnO NP was synthesized and tested against pathogenic (Staphylococcus aureus, Pseudomonas aeruginosa) and non-pathogenic (Bacillus subtilis, Escherichia coli) strains. Experimental results confirmed species-selective inhibition consistent with model predictions, and interpretable features revealed key genomic drivers of susceptibility, including oxidative stress and DNA repair pathways. By uniting AI with mechanistic insight and empirical validation, GenSelTox advances the field toward precision nanotherapeutics. The platform offers a scalable, interpretable, and generalizable framework for selective nanoparticle design, with broad potential for adaptation to antifungal, antiviral, and anticancer applications. 

![](/Figure.jpg)

**Guidelines**

This repository contains all the necessary files for screening selectevely antimicrobial inorganic NPs. The repository includes folders for `LoadingGenomicData`, `Data and Preprocessing`, `ModelBuilding`, `ModelOptimization`, `GeneticAlgorithm` and `ReinforcementLearning`.

**Loading Genomic Data**

Contains scripts for fetching genomic datasets from KEGG: a functional orthologs dataset and a metabolic pathways dataset.

**Data and Preprocessing**

Contains preprocessed datasets with genomic descriptors: one merged with functional orthologs and one merged with metabolic pathways.

**ML Model Building**

Contains code for feature selection based on correlation and variance thresholds, feature importance analysis, permutation importance, and model training.

**ML Model Optimization:**  

Top-performing models from the selection stage were further optimized via hyperparameter tuning. CatBoost and XGBoost regressors achieved the best performance after optimization and were used for predicting the minimal inhibitory concentration (MC) of NPs. This folder contains Python scripts for optimization of both models.

**Genetic Algorithm**

Contains scripts for generating selectively antimicrobial NP candidates using a genetic algorithm. The folder is organized into CatBoost and XGBoost subfolders, each containing the required datasets and Python scripts for unique compound generation, crossover, and mutation. The main script (ga_main.py) runs the evolutionary optimization for a user-defined population size and number of generations. Optimized models are used to predict antimicrobial activity of generated candidates. Top candidates selected by highest fitness score are saved to the output folder.
