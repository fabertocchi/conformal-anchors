import pandas as pd
import numpy as np
import random


# Number of patients
N_SAMPLES = 1000

# Dictonary with, for each disease and for each test used for that disease, tuples of (probability of range, lower extreme of range, upper extreme of range)
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

# To extract names of the diseases
DISEASE_NAMES = list(TESTS.keys())

# Number of diseases
N_DISEASES = len(DISEASE_NAMES)

# To extract names
TESTS_NAMES = list(set([test for disease_name in TESTS.keys() for test in TESTS[disease_name]['tests']]))

# List of probabilities of each disease (uniform probability case)
DISEASE_PROBS = [1 / N_DISEASES] * N_DISEASES

# Define generic symptoms
SYMPTOMS_NAMES = ['fever_severity', 'cough_severity', 'chest_pain_severity', 'abdominal_pain_severity', 'fatigue_level', 'nausea']

# Extract number of symptoms
N_SYMPTOMS = len(SYMPTOMS_NAMES)

def generate_disease_labels():
    """Generate disease labels with UNIFORM distribution"""

    return np.random.choice(DISEASE_NAMES, size=N_SAMPLES, p=DISEASE_PROBS)

def generate_generic_symptoms(disease_labels):
    """Generate generic symptoms based on disease patterns"""

    symptoms = np.zeros((N_SAMPLES, N_SYMPTOMS))

    for i, disease in enumerate(disease_labels):
        if TESTS[disease]['group'] == "Lung":
            # Lung diseases: higher fever, cough, chest pain, moderate fatigue
            symptoms[i, 0] = np.random.beta(3, 2) * 10          # fever_severity (0-10)
            symptoms[i, 1] = np.random.beta(4, 1.5) * 10        # cough_severity 
            symptoms[i, 2] = np.random.beta(3, 2) * 10          # chest_pain_severity
            symptoms[i, 3] = np.random.beta(1, 4) * 10          # abdominal_pain_severity (low)
            symptoms[i, 4] = np.random.beta(2.5, 2) * 10        # fatigue_level
            symptoms[i, 5] = np.random.beta(1, 3) * 10          # nausea (low)

            # RELEVANT symptoms for lung diseases (low missingness - human error/oversight)
            if np.random.random() < 0.02:   # 5% chance - data entry error
                symptoms[i, 0] = np.nan     # fever not recorded
            if np.random.random() < 0.005:  # 2% chance - rare oversight
                symptoms[i, 1] = np.nan     # cough not recorded (main symptom!)
            if np.random.random() < 0.03:   # 6% chance
                symptoms[i, 2] = np.nan     # chest_pain not recorded

            # LESS RELEVANT symptoms for lung diseases (higher missingness)
            if np.random.random() < 0.20:   # 25% chance - often not asked
                symptoms[i, 3] = np.nan     # abdominal_pain not recorded
            if np.random.random() < 0.15:   # 20% chance - often not asked
                symptoms[i, 5] = np.nan     # nausea not recorded
            
            # GENERAL symptom (moderate missingness across all diseases)
            if np.random.random() < 0.10:   # 12% chance - often forgotten
                symptoms[i, 4] = np.nan     # fatigue not recorded

        else:
            # Stomach diseases: moderate fever, low cough, high abdominal pain, nausea
            symptoms[i, 0] = np.random.beta(2, 3) * 10          # fever_severity (lower)
            symptoms[i, 1] = np.random.beta(1, 4) * 10          # cough_severity (low)
            symptoms[i, 2] = np.random.beta(1, 4) * 10          # chest_pain_severity (low)
            symptoms[i, 3] = np.random.beta(4, 1.5) * 10        # abdominal_pain_severity
            symptoms[i, 4] = np.random.beta(2.5, 2) * 10        # fatigue_level
            symptoms[i, 5] = np.random.beta(3, 2) * 10          # nausea

            # RELEVANT symptoms for stomach diseases (low missingness - human error/oversight)
            if np.random.random() < 0.005:  # 2% chance - rare oversight
                symptoms[i, 3] = np.nan     # abdominal_pain not recorded (main symptom!)
            if np.random.random() < 0.01:   # 4% chance - data entry error
                symptoms[i, 5] = np.nan     # nausea not recorded (common symptom)
            if np.random.random() < 0.03:   # 6% chance
                symptoms[i, 0] = np.nan     # fever not recorded

            # LESS RELEVANT symptoms for stomach diseases (higher missingness)
            if np.random.random() < 0.20:   # 20% chance - often not asked
                symptoms[i, 1] = np.nan     # cough not recorded
            if np.random.random() < 0.15:   # 25% chance - often not asked
                symptoms[i, 2] = np.nan     # chest_pain not recorded
            
            # GENERAL symptom (moderate missingness across all diseases)
            if np.random.random() < 0.10:   # 12% chance - often forgotten
                symptoms[i, 4] = np.nan     # fatigue not recorded
        
    # Add some noise and overlap between groups
    noise = np.random.normal(0, 0.5, symptoms.shape)
    symptoms += noise
    symptoms = np.clip(symptoms, 0, 10)  # Keep in 0-10 range
    
    return symptoms

def generate_diagnostic_tests(disease_labels):
    """Generate disease-specific diagnostic test results"""

    test_data = {}

    # Initialize all test columns with NaN
    for test_name in TESTS_NAMES:
        test_data[test_name] = np.full(N_SAMPLES, np.nan)

    for i, disease in enumerate(disease_labels):
        for test_name in TESTS[disease]['tests']:

            # Generate probability from 0 to 1 inclusive
            gen_prob = random.random()

            for (disease_prob, range_low, range_high) in TESTS[disease]['tests'][test_name]:
                
                if gen_prob <= disease_prob:
                    if type(range_low) is int:
                        test_data[test_name][i] = random.randint(range_low, range_high)
                        
                    elif type(range_low) is float:
                        test_data[test_name][i] = random.uniform(range_low, range_high)

                    else:
                        raise ValueError("Unexpected type: " + str(type(range_low)) + " for value " + str(range_low))
                    
                    break
                    
                gen_prob -= disease_prob
                
    
    return test_data

def generate_patient_data():
    """Generate complete synthetic dataset with both generic symptoms and specific test data"""

    # Generate disease labels
    disease_labels = generate_disease_labels()

    # Generate symptoms data
    symptoms = generate_generic_symptoms(disease_labels)

    # Generate specific test data
    test_data = generate_diagnostic_tests(disease_labels)

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
    
    # Add test data
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
        .groupby(df["disease_group"])  # group by disease type
        .sum()                 # sum True values (count missing)
    )
    print("\nNumber of missing generic symptoms for disease type:\n", missing_counts)
