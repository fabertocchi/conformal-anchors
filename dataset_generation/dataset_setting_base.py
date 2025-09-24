import pandas as pd
import numpy as np

def generate_disease_labels(n_samples, disease_labels, disease_probs):
    """Generate disease labels with realistic distribution (more common diseases have higher probabilities)"""

    return np.random.choice(disease_labels, size=n_samples, p=disease_probs)

def generate_generic_symptoms(n_samples, diseases, disease_labels):
    """Generate generic symptoms based on disease patterns"""

    symptoms = np.zeros((n_samples, 6))

    for i, disease in enumerate(disease_labels):
        if diseases[disease]['group'] == "Lung":
            # Lung diseases: higher fever, cough, chest pain, moderate fatigue
            symptoms[i, 0] = np.random.beta(3, 2) * 10          # fever_severity (0-10)
            symptoms[i, 1] = np.random.beta(4, 1.5) * 10        # cough_severity 
            symptoms[i, 2] = np.random.beta(3, 2) * 10          # chest_pain_severity
            symptoms[i, 3] = np.random.beta(1, 4) * 10          # abdominal_pain_severity (low)
            symptoms[i, 4] = np.random.beta(2.5, 2) * 10        # fatigue_level
            symptoms[i, 5] = np.random.beta(1, 3) * 10          # nausea (low)
        else: # stomach diseases
            # Stomach diseases: moderate fever, low cough, high abdominal pain, nausea
            symptoms[i, 0] = np.random.beta(2, 3) * 10          # fever_severity (lower)
            symptoms[i, 1] = np.random.beta(1, 4) * 10          # cough_severity (low)
            symptoms[i, 2] = np.random.beta(1, 4) * 10          # chest_pain_severity (low)
            symptoms[i, 3] = np.random.beta(4, 1.5) * 10        # abdominal_pain_severity
            symptoms[i, 4] = np.random.beta(2.5, 2) * 10        # fatigue_level
            symptoms[i, 5] = np.random.beta(3, 2) * 10          # nausea
    
    # Add some noise and overlap between groups
    noise = np.random.normal(0, 0.5, symptoms.shape)
    symptoms += noise
    symptoms = np.clip(symptoms, 0, 10)  # Keep in 0-10 range
    
    return symptoms

def generate_diagnostic_tests(disease_labels):
    """Generate disease-specific diagnostic test results"""

    # Initialize all test columns with NaN
    test_data = {}
    
    # Define all possible tests
    tests = [

        # Lung tests
        'pulmonary_function',           # FEV1% predicted (0-100%)
        'chest_xray_score',             # Chest X-ray score (0-10)
        'sputum_neutrophil_percent',    # Sputum neutrophil % (0-100%)
        'wbc_count',                    # White blood cell count (3,000-25,000 cells/µL)
        
        # Stomach tests
        'endoscopy_score',              # Endoscopy score (0-10)
        'h_pylori_level',               # H. pylori level (0-3 categorical)
        'hemoglobin',                   # Hemoglobin (5-18 g/dL)
        'gastric_ph'                    # Gastric pH (1-8)
    ]

    for test in tests:
        test_data[test] = np.full(n_samples, np.nan)

    # Generate test values based on disease
    for i, disease in enumerate(disease_labels):
        if disease == 'Bronchitis':
            test_data['pulmonary_function'][i] = np.random.beta(2, 3) * 100             # moderately reduced function
            test_data['chest_xray_score'][i] = np.random.beta(3, 2) * 10                # mild to moderate inflammation
            test_data['sputum_neutrophil_percent'][i] = np.random.beta(4, 1) * 100      # high inflammation (range 0-100)
            test_data['wbc_count'][i] = np.random.beta(2, 3) * 10000 + 8000             # elevated fighting infection
            
        elif disease == 'Copd':
            test_data['pulmonary_function'][i] = np.random.beta(1, 4) * 100             # severely reduced
            test_data['chest_xray_score'][i] = np.random.beta(4, 1) * 10                # clear abnormalities 
            test_data['sputum_neutrophil_percent'][i] = np.random.beta(2, 2) * 60 + 20  # variable
            test_data['wbc_count'][i] = np.random.beta(2, 2) * 16000 + 4000             # variable
            
        elif disease == 'Pneumonia':
            test_data['pulmonary_function'][i] = np.random.beta(3, 2) * 30 + 60         # mildly reduced
            test_data['chest_xray_score'][i] = np.random.beta(4, 1) * 10                # high (consolidation clearly visible)
            test_data['sputum_neutrophil_percent'][i] = np.random.beta(4, 1) * 20 + 80  # very high (acute bacterial infection)
            test_data['wbc_count'][i] = np.random.beta(3, 2) * 13000 + 12000            # very high (fighting acute infection)
            
        elif disease == 'Gastritis':
            test_data['endoscopy_score'][i] = np.random.beta(4, 1) * 10                                 # high inflammation visible
            test_data['h_pylori_level'][i] = np.random.choice([0, 1, 2, 3], p=[0.3, 0.25, 0.25, 0.2])   # 30% negative (0), 25% mild (1), 25% moderate (2), 20% heavy (3)
            test_data['hemoglobin'][i] = np.random.beta(2, 3) * 6 + 10                                  # normal to slightly low 
            test_data['gastric_ph'][i] = np.random.beta(2, 2) * 6 + 1                                   # variable
            
        elif disease == 'Gastric_cancer':
            test_data['endoscopy_score'][i] = np.random.beta(4, 1) * 10                                 # very high, tumor visible
            test_data['h_pylori_level'][i] = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])     # 40% negative (0), 20% each for 1,2,3
            test_data['hemoglobin'][i] = np.random.beta(1, 3) * 7 + 5                                   # low (bleeding from tumor)
            test_data['gastric_ph'][i] = np.random.beta(1, 3) * 7 + 1                                   # variable to high (loss of acid production)
            
        elif disease == 'Peptic_ulcers':
            test_data['endoscopy_score'][i] = np.random.beta(4, 1) * 10                                 # high (ulcers visible)
            test_data['h_pylori_level'][i] = np.random.choice([0, 1, 2, 3], p=[0.2, 0.2, 0.3, 0.3])     # 20% negative (0), 20% mild (1), 30% moderate (2), 30% heavy (3) 
            test_data['hemoglobin'][i] = np.random.beta(2, 3) * 8 + 8                                   # variable (depends on bleeding)
            test_data['gastric_ph'][i] = np.random.beta(1, 3) * 7 + 1                                   # low (acidic environment)
    
    return test_data

def generate_patient_data(diseases, disease_labels, disease_probs, n_samples=1000):
    """Generate complete synthetic dataset"""

    dis_labels = generate_disease_labels(n_samples, disease_labels, disease_probs)
    symptoms = generate_generic_symptoms(n_samples, diseases, dis_labels)
    test_data = generate_diagnostic_tests(dis_labels)

    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': range(1, n_samples + 1),
        'disease': dis_labels,
        'disease_group': [diseases[d]['group'] for d in dis_labels],
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

    # Number of patients
    n_samples = 1000

    # Definition of diseases and their group
    diseases = {
            'Bronchitis': {'group': 'Lung', 'id': 0},
            'Copd': {'group': 'Lung', 'id': 1}, 
            'Pneumonia': {'group': 'Lung', 'id': 2},
            'Gastritis': {'group': 'Stomach', 'id': 3},
            'Gastric_cancer': {'group': 'Stomach', 'id': 4},
            'Peptic_ulcers': {'group': 'Stomach', 'id': 5}
        }

    # Extraction of disease labels and number of diseases
    disease_labels = list(diseases.keys())
    n_diseases = len(disease_labels)

    # List of probabilities of each disease (uniform probability case)
    disease_probs = [1/n_diseases] * n_diseases

    # Generate dataset
    df = generate_patient_data(diseases, disease_labels, disease_probs, 1000)

    df.to_csv('synthetic_patient_setting_base.csv', index=False)
    
    
    # Display basic info
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDisease distribution:")
    print(df['disease'].value_counts())
    
    print("\nMissing data summary:")
    print(df.isnull().sum().sort_values(ascending=False))

