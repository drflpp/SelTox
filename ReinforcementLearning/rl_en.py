import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

import yaml
import os
import pickle
import joblib

history = list()
positive_reward = list()

SAVING_FREQUENCY = 200

class CatalystEnv(gym.Env):
    def __init__(self, df, bacteria_dict=None, bacteria_strain_1=None, bacteria_strain_2=None):
        super(CatalystEnv, self).__init__()
        self.file_path = os.path.dirname(__file__)
        
        # Dataset is already filtered; drop only the auxiliary column
        self.df = df.drop(['MIC_NP___g_mL_'], axis=1).reset_index(drop=True)
        
        # Store bacteria dictionary and parameters for the two bacteria
        self.bacteria_dict = bacteria_dict if bacteria_dict is not None else {}
        self.bacteria_strain_1 = bacteria_strain_1
        self.bacteria_strain_2 = bacteria_strain_2
        
        # Get parameters for both bacteria
        if bacteria_strain_1 and bacteria_strain_1 in self.bacteria_dict:
            params_1 = self.bacteria_dict[bacteria_strain_1]
            self.bacteria_params_1 = dict(params_1) if params_1 else {}
        else:
            self.bacteria_params_1 = {}
            if bacteria_strain_1:
                print(f"Warning: bacteria '{bacteria_strain_1}' not found in dictionary")
        
        if bacteria_strain_2 and bacteria_strain_2 in self.bacteria_dict:
            params_2 = self.bacteria_dict[bacteria_strain_2]
            self.bacteria_params_2 = dict(params_2) if params_2 else {}
        else:
            self.bacteria_params_2 = {}
            if bacteria_strain_2:
                print(f"Warning: bacteria '{bacteria_strain_2}' not found in dictionary")
        
        # Use first bacteria parameters as base (if specified)
        self.base_params = self.bacteria_params_1.copy() if self.bacteria_params_1 else {}

        self.iter_count = 0

        self.config = self.load_config()
        
        # Drop columns not needed for training
        to_drop_cols = ['bacteria', 'strain', 'bacteria_strain']
        self.df = self.df.drop([col for col in to_drop_cols if col in self.df.columns], axis=1)

        # Load CatBoost model with encoders and scaler
        model_path = os.path.join(self.file_path, 'predictors', 'model_cat_p17.joblib')
        if os.path.exists(model_path):
            model_dict = joblib.load(model_path)
            self.catboost_model = model_dict['model']
            self.encoders = model_dict['encoders']
            self.scaler = model_dict['scaler']
            self.cat_cols = model_dict['cat_cols']
            self.num_cols = model_dict['num_cols']
            self.feature_names = model_dict['feature_names']
        else:
            print(f"Warning: model not found at path {model_path}")
            self.catboost_model = None
            self.encoders = None
            self.scaler = None
            self.cat_cols = None
            self.num_cols = None
            self.feature_names = None
        
        # Build dependency dict: np_synthesis -> all dependent parameters
        self.synthesis_to_params = self.form_synthesis_dependency_dict()
        
        # Get list of all available np_synthesis with frequencies
        synthesis_counts = self.df['np_synthesis'].value_counts().sort_values()
        self.synthesis_list = synthesis_counts.index.tolist()
        synthesis_frequencies = synthesis_counts.values.tolist()
        
        # Compute cumulative probabilities for np_synthesis selection
        total_count = sum(synthesis_frequencies)
        self.synthesis_cumulative_probs = []
        cum_sum = 0
        for freq in synthesis_frequencies:
            cum_sum += freq / total_count
            self.synthesis_cumulative_probs.append(cum_sum)
        
        self.n_synthesis = len(self.synthesis_list)

        # Get bounds for numeric parameters
        self.np_size_avg_min = self.df['np_size_avg__nm_'].min()
        self.np_size_avg_max = self.df['np_size_avg__nm_'].max()
        
        # Compute max deviations of min and max from avg
        self.df['min_ratio'] = self.df['np_size_min__nm_'] / self.df['np_size_avg__nm_']
        self.df['max_ratio'] = self.df['np_size_max__nm_'] / self.df['np_size_avg__nm_']
        self.min_ratio_min = self.df['min_ratio'].min()  # Minimum ratio (usually < 1)
        self.min_ratio_max = self.df['min_ratio'].max()  # Maximum ratio
        self.max_ratio_min = self.df['max_ratio'].min()  # Minimum ratio
        self.max_ratio_max = self.df['max_ratio'].max()  # Maximum ratio
        
        # Bounds for time_set__hours_
        self.time_set_min = self.df['time_set__hours_'].min()
        self.time_set_max = self.df['time_set__hours_'].max()

        # Action space dimensionality:
        # 1. np_synthesis - selection by frequency (1 value)
        # 2. np_size_avg__nm_ - 0-1 scaling (1 value)
        # 3. np_size_max__nm_ - relative to np_size_avg__nm_ (1 value)
        # 4. np_size_min__nm_ - relative to np_size_avg__nm_ (1 value)
        # 5. time_set__hours_ - 0-1 scaling (1 value)
        
        self.action_dim = 1 + 1 + 1 + 1 + 1  # 5 parameters
        # Action space: all values in [0, 1]
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
   
        # Initialize observation_space with correct dimensionality
        # Create test state to determine dimensionality
        # Use state as it will be formed after step()
        # (simulate action_dict with base parameters)
        test_state = {}
        # Add all possible parameters from dataset (except auxiliary)
        sample_row = self.df.iloc[0].to_dict()
        for key, value in sample_row.items():
            if key not in ['bacteria', 'strain', 'bacteria_strain', 'Unnamed: 0']:
                test_state[key] = value
        # Add parameters from bacteria dict (if specified)
        for key, value in self.base_params.items():
            test_state[key] = value
        
        # Define fixed list of numeric columns for observation
        test_state_df = pd.DataFrame([test_state])
        self.observation_numeric_cols = sorted(test_state_df.select_dtypes(include=[np.number]).columns.tolist())
        obs_dim = len(self.observation_numeric_cols)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.state = None

    def form_synthesis_dependency_dict(self):
        """
        Build dependency dict: np_synthesis -> all dependent parameters.
        Dependent parameters: Source_origin, method, Solvent_for_extract, Template_type,
        Capping_type, shape, Bio_component_class, amw, chi0v, Red_env_strength,
        Red_env_type, coating, Valance_electron, Solvent_polar, lipinskiHBA,
        CrippenClogP, hallKierAlpha, kappa1, NumHBA
        """
        synthesis_to_params = {}
        
        # Get unique np_synthesis values
        unique_synthesis = self.df['np_synthesis'].unique()
        
        for synthesis in unique_synthesis:
            if pd.isna(synthesis):
                continue
                
            synthesis_df = self.df[self.df['np_synthesis'] == synthesis]
            
            # For each dependent parameter take first value
            # (assumed identical for a given np_synthesis)
            params = {}
            
            dependent_params = [
                'Source_origin', 'method', 'Solvent_for_extract', 'Template_type',
                'Capping_type', 'shape', 'Bio_component_class', 'amw', 'chi0v', 'Red_env_strength',
                'Red_env_type', 'coating', 'Valance_electron', 'Solvent_polar',
                'lipinskiHBA', 'CrippenClogP', 'hallKierAlpha', 'kappa1', 'NumHBA'
            ]
            
            for param in dependent_params:
                if param in synthesis_df.columns:
                    # Take first value (or use mode for categorical)
                    value = synthesis_df[param].iloc[0]
                    params[param] = value
                else:
                    print(f"Warning: column '{param}' not found for np_synthesis '{synthesis}'")
            
            synthesis_to_params[synthesis] = params
        
        return synthesis_to_params

    def load_config(self):
        config = dict()
        config_path = os.path.join(self.file_path, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        return config

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state with a random catalyst from the dataset.

        Args:
            seed: Seed for reproducibility.
            options: Additional options (not used).

        Returns:
            observation: Initial observation.
            info: Additional information.
        """
        # Initialize Gymnasium RNG
        super().reset(seed=seed)
        
        # Pick random row from dataset
        random_idx = self.np_random.integers(0, len(self.df))
        self.state = self.df.iloc[random_idx].to_dict()
        
        # Add parameters from bacteria dict (if specified)
        for key, value in self.base_params.items():
            self.state[key] = value
        
        # Get observation to determine dimensionality
        observation = self._get_observation()
        actual_shape = observation.shape
        
        # Update observation space from actual data
        # (if dimensionality changed or not yet set)
        if (not hasattr(self, '_obs_shape_initialized') or
            self.observation_space.shape != actual_shape):
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=actual_shape,
                dtype=np.float32
            )
            self._obs_shape_initialized = True
            print(f"Observation space initialized with shape: {actual_shape}")
        
        # Reset episode step counter
        self.steps = 0
        
        info = {
            'seed': seed,
            'dataset_index': random_idx
        }
        
        return observation, info

    def _get_observation(self):
        """Convert state to numeric vector for RL agent."""
        # Build DataFrame from state
        state_df = pd.DataFrame([self.state])
        
        # Use fixed list of numeric columns
        # Create zero vector for all columns
        observation = np.zeros(len(self.observation_numeric_cols), dtype=np.float32)
        
        # Fill with values from state (if present)
        for i, col in enumerate(self.observation_numeric_cols):
            if col in state_df.columns:
                value = state_df[col].iloc[0]
                # Ensure value is numeric and not NaN
                if pd.notna(value) and isinstance(value, (int, float, np.number)):
                    observation[i] = float(value)
                else:
                    observation[i] = 0.0
            else:
                observation[i] = 0.0
        
        return observation

    def _parse_action(self, action):
        """Convert agent action to catalyst parameters."""
        action_dict = {}
        idx = 0
        
        # 1. Select np_synthesis using frequencies
        synthesis_value = action[idx]
        # Find synthesis method index from cumulative probabilities
        synthesis_index = 0
        for i, cum_prob in enumerate(self.synthesis_cumulative_probs):
            if synthesis_value <= cum_prob:
                synthesis_index = i
                break
        else:
            synthesis_index = len(self.synthesis_list) - 1
        
        selected_synthesis = self.synthesis_list[synthesis_index]
        action_dict['np_synthesis'] = selected_synthesis
        idx += 1
        
        # 2. Set all dependent parameters from np_synthesis automatically
        if selected_synthesis in self.synthesis_to_params:
            dependent_params = self.synthesis_to_params[selected_synthesis]
            for param, value in dependent_params.items():
                action_dict[param] = value
        else:
            print(f"Warning: np_synthesis '{selected_synthesis}' not found in dependency dict")
        
        # 3. Select np_size_avg__nm_ (0-1 scaling)
        np_size_avg_value = action[idx]
        action_dict['np_size_avg__nm_'] = round(
            self.np_size_avg_min + np_size_avg_value * (self.np_size_avg_max - self.np_size_avg_min),
            2
        )
        idx += 1
        
        # 4. Select np_size_min__nm_ relative to np_size_avg__nm_
        # 0 -> max deviation from mean, 1 -> close to mean (but not equal)
        np_size_min_value = action[idx]
        # Compute ratio: from min_ratio_min to 0.99 (cannot equal avg)
        # Ensure min_ratio >= 0.01 so np_size_min is not 0
        min_ratio_lower_bound = max(self.min_ratio_min, 0.01)
        min_ratio = min_ratio_lower_bound + np_size_min_value * (0.99 - min_ratio_lower_bound)
        action_dict['np_size_min__nm_'] = round(action_dict['np_size_avg__nm_'] * min_ratio, 2)
        
        idx += 1
        
        # 5. Select np_size_max__nm_ from difference (avg - min)
        # Formula: max = avg + k * (avg - min), k in [0.75, 1.25]
        # k = 0.75 -> (max - avg) smaller than (avg - min)
        # k = 1.0 -> symmetric: (max - avg) = (avg - min)
        # k = 1.25 -> (max - avg) larger than (avg - min)
        np_size_max_value = action[idx]
        diff = action_dict['np_size_avg__nm_'] - action_dict['np_size_min__nm_']
        k = 0.75 + np_size_max_value * (1.25 - 0.75)  # k in [0.75, 1.25]
        action_dict['np_size_max__nm_'] = round(action_dict['np_size_avg__nm_'] + k * diff, 2)
        idx += 1
        
        # 6. Select time_set__hours_ (0-1 scaling)
        time_set_value = action[idx]
        action_dict['time_set__hours_'] = round(
            self.time_set_min + time_set_value * (self.time_set_max - self.time_set_min),
            2
        )
        idx += 1
        
        # Copy for history WITHOUT bacteria params (names only)
        history_dict = action_dict.copy()
        
        # Add only bacteria names to history (not params)
        if self.bacteria_strain_1:
            history_dict['bacteria_strain_1'] = self.bacteria_strain_1
        if self.bacteria_strain_2:
            history_dict['bacteria_strain_2'] = self.bacteria_strain_2
        
        # Save to history only main params + bacteria names
        history.append(history_dict)
        
        # Add bacteria dict params to action_dict for calculations
        for key, value in self.base_params.items():
            action_dict[key] = value
        
        return action_dict
    
    def _calculate_penalties(self, state):
        """Compute penalties for constraint violations."""
        penalty = 0
        
        # Check size order: min <= avg <= max
        if not (state['np_size_min__nm_'] <= state['np_size_avg__nm_'] <= state['np_size_max__nm_']):
            penalty -= 50
        
        return penalty
    
    def save_generation(self):
        self.iter_count += 1
        if self.iter_count % SAVING_FREQUENCY == 0:
            df_to_save = pd.DataFrame(history[-100:])
            df_to_save.to_csv('test.csv')

    def step(self, action):
        # Convert agent action to catalyst parameters
        action_dict = self._parse_action(action)
        
        # Update state (create if not yet initialized)
        if self.state is None:
            self.state = {}
        
        for key, value in action_dict.items():
            self.state[key] = value
        
        # Compute reward and predictions for both bacteria
        reward, fe, fe_bacteria_1, fe_bacteria_2 = self._calculate_reward()
        # print(reward, fe)

        # Update history with predictions for both bacteria
        update_dict = {'reward': reward, 'fe': fe}
        if fe_bacteria_1 is not None:
            update_dict['fe_bacteria_1'] = fe_bacteria_1
        if fe_bacteria_2 is not None:
            update_dict['fe_bacteria_2'] = fe_bacteria_2
        
        history[-1].update(update_dict)

        self.save_generation()

        # Check episode termination conditions
        terminated  = False
        info: dict = {'FE': fe}
        
        if fe >= 80:  # Example threshold
            terminated  = True
            info['termination_reason'] = 'FE threshold reached'
            
        truncated = False
        self.steps += 1
        if self.steps >= 1000:  # Example max episode length
            truncated = True
            info['termination_reason'] = 'max steps reached'

        return self._get_observation(), reward, terminated, truncated, info
    
    def _preprocess_for_prediction(self, state_dict):
        """
        Preprocess data for CatBoost prediction:
        1. Drop unneeded columns
        2. Apply encoders to categorical features
        3. Apply scaler to numeric features
        4. Combine features in correct order
        """
        if (self.catboost_model is None or self.encoders is None or
            self.scaler is None or self.cat_cols is None or
            self.num_cols is None or self.feature_names is None):
            return None
        
        # Build DataFrame from state dict
        data_df = pd.DataFrame([state_dict])
        
        # Drop unneeded columns
        cols_to_drop = ['np_synthesis', 'bacteria_strain', 'min_ratio', 'max_ratio', 'Unnamed: 0']
        data_df = data_df.drop([col for col in cols_to_drop if col in data_df.columns], axis=1)
        
        # Copy for processing
        data_for_prediction = data_df.copy()
        
        # Apply encoders to categorical features
        for col in self.cat_cols:
            if col in data_for_prediction.columns:
                # Convert column to object first to avoid type issues
                data_for_prediction[col] = data_for_prediction[col].astype('object')
                
                # Process values before applying encoder
                value = data_for_prediction.loc[0, col]
                
                # Convert None, NaN and other invalid values
                if pd.isna(value) or value is None:
                    # Use first dataset value for this column as default
                    if col in self.df.columns:
                        default_value = self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else ''
                    else:
                        default_value = ''
                    data_for_prediction.loc[0, col] = default_value
                    value = default_value
                
                # Convert to string for categorical features
                if not isinstance(value, str):
                    value = str(value)
                    data_for_prediction.loc[0, col] = value
                
                encoder = self.encoders[col]
                # Convert to 2D array for encoder
                try:
                    # Ensure column has correct type (string)
                    data_for_prediction[col] = data_for_prediction[col].astype(str)
                    encoded_values = encoder.transform(data_for_prediction[[col]])
                    data_for_prediction[col] = encoded_values.flatten()
                except Exception as e:
                    # If encoder cannot handle value, use default
                    print(f"Warning: error encoding column '{col}': {e}, value: {value}")
                    # Try using first value from dataset
                    if col in self.df.columns:
                        default_value = self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else ''
                        data_for_prediction[col] = data_for_prediction[col].astype('object')
                        data_for_prediction.loc[0, col] = str(default_value)
                        data_for_prediction[col] = data_for_prediction[col].astype(str)
                        encoded_values = encoder.transform(data_for_prediction[[col]])
                        data_for_prediction[col] = encoded_values.flatten()
                    else:
                        # If column missing, fill with zero
                        data_for_prediction[col] = 0
            else:
                # If column missing, add it with default value
                if col in self.df.columns:
                    default_value = self.df[col].dropna().iloc[0] if len(self.df[col].dropna()) > 0 else ''
                    data_for_prediction[col] = [str(default_value)]
                    encoder = self.encoders[col]
                    encoded_values = encoder.transform(data_for_prediction[[col]])
                    data_for_prediction[col] = encoded_values.flatten()
                else:
                    # If column not in dataset, fill with zero
                    data_for_prediction[col] = 0
        
        # Split into numeric and categorical features
        # Use DataFrame for numeric features to preserve column names
        X_num_df = data_for_prediction[self.num_cols]
        X_cat = data_for_prediction[self.cat_cols].values
        
        # Apply scaler to numeric features (pass DataFrame with column names)
        X_num_scaled = self.scaler.transform(X_num_df)
        
        # Combine features in correct order: numeric first, then categorical
        X_combined = np.hstack([X_num_scaled.astype(np.float32), X_cat.astype(np.float32)])
        
        # Build DataFrame with correct feature order
        X_final = pd.DataFrame(X_combined, columns=self.feature_names)
        
        return X_final
    
    def _calculate_reward(self):
        # Convert state (dict) for CatBoost model
        if self.state is None:
            return 0.0, 0.0, None, None
        state_dict = dict(self.state) if self.state else {}
        
        # If two bacteria specified, use comparison mechanism
        if self.bacteria_strain_1 and self.bacteria_strain_2 and self.bacteria_params_1 and self.bacteria_params_2:
            # Build state for first bacteria
            state_dict_1 = state_dict.copy()
            # Replace params with first bacteria params
            for param, value in self.bacteria_params_1.items():
                state_dict_1[param] = value
            
            # Preprocess data for first bacteria
            X_final_1 = self._preprocess_for_prediction(state_dict_1)
            
            # Get FE prediction for first bacteria
            if X_final_1 is not None and self.catboost_model is not None:
                tox_value_1 = float(self.catboost_model.predict(X_final_1)[0])
            else:
                tox_value_1 = np.random.randint(10, 100)
            
            # Build state for second bacteria
            state_dict_2 = state_dict.copy()
            # Replace params with second bacteria params
            for param, value in self.bacteria_params_2.items():
                state_dict_2[param] = value
            
            # Preprocess data for second bacteria
            X_final_2 = self._preprocess_for_prediction(state_dict_2)
            
            # Get FE prediction for second bacteria
            if X_final_2 is not None and self.catboost_model is not None:
                tox_value_2 = float(self.catboost_model.predict(X_final_2)[0])
            else:
                tox_value_2 = np.random.randint(10, 100)
            
            # Compute difference between the two values
            tox_diff = tox_value_1 - tox_value_2
            
            # Add penalties for constraint violations
            penalty = self._calculate_penalties(self.state)
            
            # Return: reward, fe (difference), fe_bacteria_1, fe_bacteria_2
            return round(tox_diff + penalty, 2), round(tox_diff, 2), round(tox_value_1, 2), round(tox_value_2, 2)
        else:
            # Standard mode (no bacteria comparison)
            # Preprocess data
            X_final = self._preprocess_for_prediction(state_dict)
            
            # Get FE prediction directly from CatBoost model
            if X_final is not None and self.catboost_model is not None:
                tox_value = float(self.catboost_model.predict(X_final)[0])
            else:
                tox_value = np.random.randint(10, 100)
            
            # Add penalties for constraint violations
            penalty = self._calculate_penalties(self.state)
            
            # Return: reward, fe, None, None (no second bacteria)
            return round(tox_value + penalty, 2), round(tox_value, 2), None, None
    
# Load data
df = pd.read_csv('final_df1_catboost_orig_bact.csv', sep=';')

# Check and filter data: drop rows where np_size_min <= np_size_avg <= np_size_max fails
print(f"\n{'='*80}")
print(f"CHECK CONDITION: np_size_min__nm_ <= np_size_avg__nm_ <= np_size_max__nm_")
print(f"{'='*80}")
print(f"Total rows in dataset before filtering: {len(df)}")

# Check condition min <= avg <= max
condition = (df['np_size_min__nm_'] <= df['np_size_avg__nm_']) & (df['np_size_avg__nm_'] <= df['np_size_max__nm_'])
violations = df[~condition]

print(f"Rows with condition violation: {len(violations)}")

if len(violations) > 0:
    print(f"Violation percentage: {len(violations) / len(df) * 100:.2f}%")
    # Drop rows with violations
    df = df[condition].reset_index(drop=True)
    print(f"Violating rows removed")

print(f"Total rows after filtering: {len(df)}")
print(f"{'='*80}\n")

# Build bacteria dictionary from bacteria_strain column
def form_bacteria_dict(df):
    """
    Build dictionary of bacteria with their parameters from bacteria_strain column.

    Args:
        df: DataFrame with data.

    Returns:
        dict: Dictionary where key is bacteria_strain, value is dict of parameters.
    """
    bacteria_dict = {}
    
    # Parameters to extract for each bacteria (updated list)
    param_columns = [
        'K01191', 'K13566', 'K07484', 'K25602', 'K11206', 'K07486', 'K12942',
        'K01153', 'K02027', 'K00849', 'K01878', 'K00432', 'K01026', 'K10844',
        'K03741', 'K00252', 'K01190', 'K03703', 'K09936', 'K07485', 'K07778',
        'K16148', 'sec_habitat', 'bac_type', 'K23945', 'common_environment',
        'K03629', 'min_Incub_period__h', 'K07050', 'K20345', 'avg_Incub_period__h',
        'K07123', 'max_Incub_period__h', 'mdr', 'prim_specific_habitat'
    ]
    
    # Check for bacteria_strain column
    if 'bacteria_strain' not in df.columns:
        print("Warning: column 'bacteria_strain' not found in dataset")
        return bacteria_dict
    
    # Get unique bacteria_strain values
    unique_bacteria = df['bacteria_strain'].unique()
    
    for bacteria in unique_bacteria:
        if pd.isna(bacteria):
            continue
            
        # Filter data for this bacteria
        bacteria_df = df[df['bacteria_strain'] == bacteria]
        
        # Build param dict for this bacteria
        bacteria_params = {}
        
        for param in param_columns:
            if param in bacteria_df.columns:
                # Take first value (assumed same for one bacteria)
                value = bacteria_df[param].iloc[0]
                bacteria_params[param] = value
            else:
                print(f"Warning: column '{param}' not found for bacteria {bacteria}")
        
        bacteria_dict[bacteria] = bacteria_params
    
    return bacteria_dict

# Build bacteria dictionary
bacteria_dict = form_bacteria_dict(df)
print(f"Built dictionary for {len(bacteria_dict)} bacteria")
print("Sample entry:", list(bacteria_dict.items())[0] if bacteria_dict else "Dictionary is empty")

# Select two bacteria for comparison
# Example: take first two available bacteria from dict
if len(bacteria_dict) >= 2:
    bacteria_list = list(bacteria_dict.keys())
    bacteria_strain_1 = "Bacillus subtilis nan"
    bacteria_strain_2 = "Pseudomonas aeruginosa nan"
    print(f"\nBacteria selected for comparison:")
    print(f"  Bacteria 1: {bacteria_strain_1}")
    print(f"  Bacteria 2: {bacteria_strain_2}")
    print(f"  Bacteria 1 params: {bacteria_dict[bacteria_strain_1]}")
    print(f"  Bacteria 2 params: {bacteria_dict[bacteria_strain_2]}")
else:
    bacteria_strain_1 = None
    bacteria_strain_2 = None
    print("\nWarning: not enough bacteria for comparison (minimum 2 required)")

# Create environment with two bacteria (params from bacteria dict)
env = CatalystEnv(df, bacteria_dict=bacteria_dict,
                  bacteria_strain_1=bacteria_strain_1, bacteria_strain_2=bacteria_strain_2)

# Environment check (optional)
check_env(env)

# Wrap environment for Stable Baselines3 compatibility
# Create new environment for wrapper (do not reuse already wrapped)
env_wrapped = DummyVecEnv([lambda: CatalystEnv(df, bacteria_dict=bacteria_dict,
                                                  bacteria_strain_1=bacteria_strain_1,
                                                  bacteria_strain_2=bacteria_strain_2)])
env = env_wrapped

# Select and initialize algorithm
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=256,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5
)


# Train agent
model.learn(total_timesteps=5120)

# Save model
# model.save("catalyst_optimizer")

df_to_save = pd.DataFrame(history)
# df_to_save = df_to_save[df_to_save['reward'] >0]
df_to_save.to_csv('result.csv')

# Load model (for later use)
# model = PPO.load("catalyst_optimizer")
