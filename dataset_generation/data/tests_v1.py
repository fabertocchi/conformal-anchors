"""Version 1: 3 diseases per macro-group, 4 tests per disease"""

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