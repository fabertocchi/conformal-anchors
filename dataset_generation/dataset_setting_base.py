import pandas as pd
import numpy as np
import random
import math
from scipy.stats import truncnorm

# Number of patients
N_SAMPLES = 1000

# For each disease, we define:
#   -'group': a high level category indicating the organ/system affected ('Lung' or 'Stomach')
#   -'tests': a dictionary mapping test names to a list of "bands", where each band is defined as (probability, lower bound, upper bound)
# Probabilities for a disease/test must sum to 1.
# We assume a uniform distribution inside each band when generating values.
TESTS = {
    'Bronchitis': {
        'group': 'Lung', 
        'tests': {
            'pulmonary_function': [
                (0.65, 70, 100),
                (0.15, 60, 69),
                (0.10, 50, 59),
                (0.05, 40, 49),
                (0.03, 30, 39),
                (0.02, 15, 29)
                # no values between 0 and 14 because are too unrealistic
            ],
            'chest_xray_score': [
                (0.45, 0, 0),
                (0.35, 1, 2),
                (0.15, 3, 4),
                (0.04, 5, 7),
                (0.01, 8, 10),
            ],
            'sputum_neutrophil_percent':[
                (0.05, 20, 39),
                (0.20, 40, 54),
                (0.35, 55, 69),
                (0.30, 70, 84),
                (0.10, 85, 100)
            ],
            'wbc_count': [
                (0.10, 3000, 6000),
                (0.35, 6001, 9000),
                (0.30, 9001, 12000),
                (0.18, 12001, 16000),
                (0.05, 16001, 20000),
                (0.02, 20001, 25000)
            ]
        }
    },
    'Copd': {
        'group': 'Lung',
        'tests': {
            'pulmonary_function': [
                (0.05, 70, 100),
                (0.07, 60, 69),
                (0.13, 50, 59),
                (0.20, 40, 49),
                (0.25, 30, 39),
                (0.30, 15, 29)
                # no values between 0 and 14 because are too unrealistic
            ],
            'chest_xray_score': [
                (0.05, 0, 0),
                (0.20, 1, 2),
                (0.35, 3, 4),
                (0.30, 5, 7),
                (0.10, 8, 10)
            ],
            'sputum_neutrophil_percent': [
                (0.25, 20, 39),
                (0.35, 40, 54),
                (0.25, 55, 69),
                (0.10, 70, 84),
                (0.05, 85, 100)
            ],
            'wbc_count': [
                (0.20, 3000, 6000),
                (0.45, 6001, 9000),
                (0.25, 9001, 12000),
                (0.08, 12001, 16000),
                (0.019, 16001, 20000),
                (0.001, 20001, 25000)
            ]
        }
    },
    'Pneumonia': {
        'group': 'Lung',
        'tests': {
            'pulmonary_function': [
                (0.25, 70, 100),
                (0.25, 60, 69),
                (0.20, 50, 59),
                (0.15, 40, 49),
                (0.10, 30, 39),
                (0.05, 15, 29)
                # no values between 0 and 14 because are too unrealistic
            ],
            'chest_xray_score': [
                (0.05, 0, 0),
                (0.15, 1, 2),
                (0.25, 3, 4),
                (0.35, 5, 7),
                (0.20, 8, 10)
            ],
            'sputum_neutrophil_percent': [
                (0.02, 20, 39),
                (0.08, 40, 54),
                (0.25, 55, 69),
                (0.35, 70, 84),
                (0.30, 85, 100)

            ],
            'wbc_count': [
                (0.03, 3000, 6000),
                (0.10, 6001, 9000),
                (0.20, 9001, 12000),
                (0.30, 12001, 16000),
                (0.22, 16001, 20000),
                (0.15, 20001, 25000)
            ]
        }
    },
    'Gastritis': {
        'group': 'Stomach',
        'tests': {
            'endoscopy_score': [
                (0.10, 0, 0),
                (0.35, 1, 2),
                (0.30, 3, 4),
                (0.15, 5, 6),
                (0.07, 7, 8),
                (0.03, 9, 10)
            ],
            'h_pylori_level': [
                (0.10, 0, 0),
                (0.15, 1, 1),
                (0.30, 2, 2),
                (0.45, 3, 3)
            ],
            'hemoglobin': [
                (0.01, 5, 8),
                (0.03, 8.1, 10),
                (0.12, 10.1, 12),
                (0.40, 12.1, 14),
                (0.35, 14.1, 16),
                (0.09, 16.1, 18)
            ],
            'gastric_ph': [
                (0.55, 1, 2),
                (0.25, 3, 3),
                (0.10, 4, 4),
                (0.06, 5, 5),
                (0.03, 6, 7),
                (0.01, 8, 8)
            ]
        }
    },
    'Gastric_cancer': {
        'group': 'Stomach',
        'tests': {
            'endoscopy_score': [
                (0.00, 0, 0),
                (0.02, 1, 2),
                (0.10, 3, 4),
                (0.20, 5, 6),
                (0.35, 7, 8),
                (0.33, 9, 10)
            ],
            'h_pylori_level': [
                (0.25, 0, 0),
                (0.30, 1, 1),
                (0.25, 2, 2),
                (0.20, 3, 3)
            ],
            'hemoglobin': [
                (0.05, 5, 8),
                (0.10, 8.1, 10),
                (0.25, 10.1, 12),
                (0.35, 12.1, 14),
                (0.20, 14.1, 16),
                (0.05, 16.1, 18)
            ],
            'gastric_ph': [
                (0.30, 1, 2),
                (0.25, 3, 3),
                (0.20, 4, 4),
                (0.15, 5, 5),
                (0.07, 6, 7),
                (0.03, 8, 8)
            ]
        }
    },
    'Peptic_ulcers': {
        'group': 'Stomach',
        'tests': {
            'endoscopy_score': [
                (0.02, 0, 0),
                (0.05, 1, 2),
                (0.20, 3, 4),
                (0.40, 5, 6),
                (0.25, 7, 8),
                (0.08, 9, 10)
            ],
            'h_pylori_level': [
                (0.15, 0, 0),
                (0.15, 1, 1),
                (0.30, 2, 2),
                (0.40, 3, 3)
            ],
            'hemoglobin': [
                (0.08, 5, 8),
                (0.15, 8.1, 10),
                (0.25, 10.1, 12),
                (0.30, 12.1, 14),
                (0.18, 14.1, 16),
                (0.04, 16.1, 18)

            ],
            'gastric_ph': [
                (0.65, 1, 2),
                (0.20, 3, 3),
                (0.08, 4, 4),
                (0.04, 5, 5),
                (0.02, 6, 7),
                (0.01, 8, 8)
            ]
        }
    }
}

# List of disease names
DISEASE_NAMES = list(TESTS.keys())

# Number of diseases
N_DISEASES = len(DISEASE_NAMES)

# List of probabilities for generating each disease label (uniform probability case)
DISEASE_PROBS = [1 / N_DISEASES] * N_DISEASES

# List of test names
TESTS_NAMES = list(set([test for disease_name in TESTS.keys() for test in TESTS[disease_name]['tests']]))

# Definition of generic symptom features
SYMPTOMS_NAMES = [
    'fever_severity', 'cough_severity', 'chest_pain_severity', 
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

# Number of symptoms
N_SYMPTOMS = len(SYMPTOMS_NAMES)

# Definition of possible range values for each test: (minimum, maximum)
TEST_BOUNDS = {
    'pulmonary_function': (0, 100),
    'chest_xray_score': (0, 10),
    'sputum_neutrophil_percent': (0, 100),
    'wbc_count': (3000, 25000),
    'endoscopy_score': (0, 10),
    'h_pylori_level': (0, 3),      
    'hemoglobin': (5, 18),
    'gastric_ph': (1, 8)
}

# Tests treated as ordinal variables
# We do not sample these tests from a truncated normal distribution, because the scores are discrete or ordinal by nature
# but we sample them from the categorical distribution defined by the bands
ORDINAL_TESTS = {
    'chest_xray_score',
    'endoscopy_score',
    'h_pylori_level'
}

def compute_disease_test_mean_variance(table):
    """
    Compute mixture mean and total variance for one (disease, test) table.
    Each table consists of bands defined as (probability, range_low, range_high), 
    representing the probability of a value falling in that band and the bounds of the band.
    Returns:
        mean: overall expected value of the test for this disease
        var: overall variance of the test for this disease
    """

    # Ensure that the sum of the probabilities of all bands is 1
    total_p = 0
    tolerance = 1e-8
    for prob, _, _ in table:
        total_p += prob
    if abs(total_p - 1.0) > tolerance:
            print(total_p)
            raise ValueError("Probabilities must sum to 1.")

    # Compute band-specific means and variances with the assumption that values are uniformly distributed within each band
    band_means = []
    band_vars = []
    
    for prob, range_low, range_high in table:
        # Mean of a uniform distribution: midpoint of range
        mid_point = (range_low + range_high) * 0.5
        # Variance of a uniform distribution: (b-a)^2 / 12
        s2 = (range_high - range_low) ** 2 / 12.0

        band_means.append((prob, mid_point))    # keeps track of probability, bin mean
        band_vars.append((prob, s2))            # keeps track of probability, bin variance
        
    # Compute overall mean using law of total expectation
    mean = sum(prob * mid_point for prob, mid_point in band_means)
    # Compute overall variance using law of total variance
    # - within: average of variances within each band
    # - between: variance of the band means relative to overall mean
    within = sum(prob * s2 for prob, s2 in band_vars)
    between = sum(prob * (mid_point - mean) ** 2 for prob, mid_point in band_means)
    var = within + between

    return mean, var

def build_stats(tables):
    """
    Build a summary statistics dictionary for all diseases and tests.

    For each disease and each test, compute:
        - Mean of the test values 
        - Variance of the test values
        - Keep the original table of bands and probabilities

    Parameters:
        tables: The input dictionary defining diseases, their groups, and test bands.

    Returns:
        stats: A nested dictionary:
                stats[disease]['group'] -> group name ('Lung' or 'Stomach')
                stats[disease]['tests'][test_name] -> dict with keys:
                    - 'table': original band table
                    - 'mean': mean of the test for this disease
                    - 'variance': variance of the test for this disease
    """
    
    # Initialize the stats dictionary
    stats = {}

    for disease, info in tables.items():
        # Create disease entry with group and empty test dictionary
        stats[disease] = {
            'group': info['group'],
            'tests': {}
        }
        for test_name, table in info['tests'].items():
            # Compute mean and variance for this test and disease
            mean, var = compute_disease_test_mean_variance(table)
            # Store results along with the original table
            entry = {
                'table': table,
                'mean': mean,
                'variance': var
            }
            # Add to stats
            stats[disease]['tests'][test_name] = entry

    # Return the fully populated dictionary with summary statistics
    return stats

# Compute summary statistics for all diseases and tests
TEST_STATS = build_stats(TESTS)

def generate_disease_labels():
    """
    Generate disease labels with UNIFORM distribution.
    Returns:
        np.array of shape (N_SAMPLES,) containing disease names for each patient
    """

    return np.random.choice(DISEASE_NAMES, size=N_SAMPLES, p=DISEASE_PROBS)

def generate_generic_symptoms(disease_labels):
    """
    Generate continuous generic symptom severities based on disease patterns.
    - Uses Beta distributions with realistic group-specific shape parameters.
    - Adds controlled missingness: higher for less clinically relevant symptoms.
    - Adds small Gaussian noise to introduce variability and overlap between groups.
    - Clips values to [0, 10] to respect symptom scale.

    Parameters:
        disease_labels: array-like
            Disease assigned to each patient
    
    Returns:
        symptoms: np.array of shape (N_SAMPLES, N_SYMPTOMS)
            Numeric symptom values with possible NaNs representing missing data
    """

    # Initialize empty array to hold symptom values
    symptoms = np.zeros((N_SAMPLES, N_SYMPTOMS))

    for i, disease in enumerate(disease_labels):
        if TESTS[disease]['group'] == "Lung":
            # Lung diseases: higher fever/cough/chest pain, moderate fatigue
            symptoms[i, 0] = np.random.beta(3, 2) * 10          # fever_severity (0-10)
            symptoms[i, 1] = np.random.beta(4, 1.5) * 10        # cough_severity 
            symptoms[i, 2] = np.random.beta(3, 2) * 10          # chest_pain_severity
            symptoms[i, 3] = np.random.beta(1, 4) * 10          # abdominal_pain_severity (low)
            symptoms[i, 4] = np.random.beta(2.5, 2) * 10        # fatigue_level
            symptoms[i, 5] = np.random.beta(1, 3) * 10          # nausea (low)

            # Introduce missing values with probabilities reflecting clinical relevance
            # Core lung symptoms (fever, cough, chest pain) are less likely missing
            if np.random.random() < 0.02: symptoms[i, 0] = np.nan     # fever 
            if np.random.random() < 0.005: symptoms[i, 1] = np.nan    # cough 
            if np.random.random() < 0.03: symptoms[i, 2] = np.nan     # chest_pain 
            # Less relevant symptoms more likely missing
            if np.random.random() < 0.20: symptoms[i, 3] = np.nan     # abdominal_pain 
            if np.random.random() < 0.15: symptoms[i, 5] = np.nan     # nausea 
            if np.random.random() < 0.10: symptoms[i, 4] = np.nan     # fatigue (general)

        else:
            # Stomach diseases: moderate fever, low cough, high abdominal pain/nausea
            symptoms[i, 0] = np.random.beta(2, 3) * 10          # fever_severity (lower)
            symptoms[i, 1] = np.random.beta(1, 4) * 10          # cough_severity (low)
            symptoms[i, 2] = np.random.beta(1, 4) * 10          # chest_pain_severity (low)
            symptoms[i, 3] = np.random.beta(4, 1.5) * 10        # abdominal_pain_severity
            symptoms[i, 4] = np.random.beta(2.5, 2) * 10        # fatigue_level
            symptoms[i, 5] = np.random.beta(3, 2) * 10          # nausea

            # Missingness patterns inverted relative to lung (abdominal and nausea more crucial)
            if np.random.random() < 0.005: symptoms[i, 3] = np.nan     # abdominal_pain (main symptom)
            if np.random.random() < 0.01: symptoms[i, 5] = np.nan      # nausea (relevant)
            if np.random.random() < 0.03: symptoms[i, 0] = np.nan      # fever 
            if np.random.random() < 0.20: symptoms[i, 1] = np.nan      # cough (irrelevant)
            if np.random.random() < 0.15: symptoms[i, 2] = np.nan      # chest_pain (less relevant)
            if np.random.random() < 0.10: symptoms[i, 4] = np.nan      # fatigue (general)
        
    # Add some noise and overlap between groups
    noise = np.random.normal(0, 0.5, symptoms.shape)
    symptoms += noise
    symptoms = np.clip(symptoms, 0, 10)  # Keep in 0-10 range
    
    return symptoms

def sample_truncated_normal(mean, variance, lower, upper):
    """
    Sample one value from N(mean, variance) truncated to [lower, upper].
    Used for continuous diagnostic tests where values must remain within realistic bounds..

    Parameters:
        mean (float): mean of the normal distribution
        variance (float): variance of the normal distribution
        lower (float or None): lower bound (inclusive); if None, no lower bound
        upper (float or None): upper bound (inclusive); if None, no upper bound

    Returns:
        float: a sample from the truncated normal distribution
    """

    if variance <= 0:
        # Degenerate variance: return mean clipped to bounds
        return min(max(mean, lower), upper)

    # Standard deviation
    sd = math.sqrt(variance)

    # Compute standardized bounds
    a = (lower - mean) / sd if lower is not None else -math.inf
    b = (upper - mean) / sd if upper is not None else math.inf
    
    return truncnorm.rvs(a, b, loc=mean, scale=sd)

def generate_categorical_diagnostic_tests(table):
    """
    Sample a value from a categorical distribution defined by bands.
    Steps:
        1. Pick band based on its probability.
        2. Sample uniformly inside that band.
    Used for ordinal diagnostic tests.

    Parameters:
    - table: list of tuples [(probability, range_low, range_high), ...]
      Each tuple defines a band:
        - probability: probability of selecting this band
        - range_low: lower bound of values in the band
        - range_high: upper bound of values in the band
    
    Returns:
    - value: a numeric value sampled according to the table
    """

    # Generate a random number from 0 to 1 inclusive: it determines which band will be selected based on cumulative probability
    r = random.random()

    for disease_prob, range_low, range_high in table:
        # Check if the random number falls within the probability of this band
        if r <= disease_prob:
            # Sample uniformly inside the selected band
            if type(range_low) is int:
                value = random.randint(range_low, range_high)
            elif type(range_low) is float:
                value = random.uniform(range_low, range_high)
            else:
                # Raise an error if type is unexpected
                raise ValueError("Unexpected type: " + str(type(range_low)) + " for value " + str(range_low))
            
            # Once a value is sampled, exit the loop
            break
            
        # Adjust the random number for the next band, effectively mapping r into the remaining probability space
        r -= disease_prob
                
    return value

def generate_diagnostic_tests_values(disease_labels):
    """
    Generate diagnostic test values for each patient given their disease.
    - Continuous tests -> truncated normal with disease-specific mean/var.
    - Ordinal tests -> categorical sampling from probability table.

    Parameters:
        disease_labels: array-like of shape (N_SAMPLES,)
            Disease assigned to each patient

    Returns:
        test_data: dict
            Keys = test names, values = np.array of length N_SAMPLES
            Each array contains the synthetic test values for that test
    """
   
    test_data = {}

    # Initialize all test columns with NaN
    for test_name in TESTS_NAMES:
        # Each key is a test name, each value is an array of NaNs (will fill per patient)
        test_data[test_name] = np.full(N_SAMPLES, np.nan)

    for i, disease in enumerate(disease_labels):
        # Ensure the disease exists in the stats dictionary
        if disease not in TEST_STATS:
            raise KeyError(f"Disease '{disease}' not found in TEST_STATS.")
        # Iterate over all tests defined for this disease
        for test_name, info in TEST_STATS[disease]['tests'].items():
            table = info['table']       # original band probabilities and ranges
            mean = info['mean']         # mean of the test for this disease
            var = info['variance']      # variance of the test for this disease

            # Decide sampling method
            if (test_name in ORDINAL_TESTS):
                # Ordinal test: pick a band based on probability, then sample uniformly inside band
                value = generate_categorical_diagnostic_tests(table)
            else:
                # Continuous test: truncated normal sample within realistic bounds
                lower, upper = TEST_BOUNDS.get(test_name, (None, None))
                value = sample_truncated_normal(mean, var, lower, upper)

            # Store sampled value for this patient and test
            test_data[test_name][i] = value

    return test_data

def generate_patient_data():
    """
    Generate full synthetic patient dataset, which includes:
      - disease labels (assigned uniformly across diseases)
      - generic symptoms (continuous, disease-group dependent)
      - diagnostic tests (continuous or ordinal, conditional on disease)
    
    Returns:
        df: pandas.DataFrame
            Each row corresponds to a patient, with columns:
                - patient_id: unique identifier
                - disease: assigned disease label
                - disease_group: high-level group (Lung/Stomach)
                - Generic symptoms: fever, cough, chest pain, abdominal pain, fatigue, nausea
                - Diagnostic tests: pulmonary_function, chest_xray_score, etc.
    """

    # Generate disease labels (uniformly)
    disease_labels = generate_disease_labels()

    # Generate generic symptoms data
    symptoms = generate_generic_symptoms(disease_labels)

    # Generate diagnostic test data
    test_data = generate_diagnostic_tests_values(disease_labels)

    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': range(1, N_SAMPLES + 1),
        'disease': disease_labels,
        'disease_group': [TESTS[d]['group'] for d in disease_labels],
        'fever_severity': symptoms[:, 0],
        'cough_severity': symptoms[:, 1], 
        'chest_pain_severity': symptoms[:, 2],
        'abdominal_pain_severity': symptoms[:, 3],
        'fatigue_level': symptoms[:, 4],
        'nausea': symptoms[:, 5]
    })
    
    # Add diagnostic test columns
    for test_name, test_values in test_data.items():
        df[test_name] = test_values

    return df


if __name__ == "__main__":

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Generate dataset
    df = generate_patient_data()
    df.to_csv('synthetic_patient_setting_base_new.csv', index=False)
    
    # Display basic info
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDisease distribution:")
    print(df['disease'].value_counts())
    
    print("\nMissing data summary:")
    print(df.isnull().sum().sort_values(ascending=False))

    
    missing_counts = (
        df[SYMPTOMS_NAMES].isna()                # True where missing
        .groupby(df["disease_group"])            # group by disease type
        .sum()                 
    )
    print("\nNumber of missing generic symptoms for disease type:\n", missing_counts)
