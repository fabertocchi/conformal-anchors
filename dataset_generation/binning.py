import pandas as pd
import numpy as np
from data.tests_v2 import TESTS
# from data.tests_v1 import TESTS

# Upload the two versions of the dataset
df_v1 = pd.read_csv("./synthetic_patient_basic_setting_v1.csv")
df_v2 = pd.read_csv("./synthetic_patient_basic_setting_v2.csv")

# Define X and y
y = df_v2['disease']
X = df_v2.drop(columns=['disease', 'patient_id', 'disease_group'])

# Bin the generic symptoms columns into 11 bins: 0,1,...,10
df_v2_binned = df_v2.copy()

generic_symptoms_cols = ['fever_severity', 'cough_severity', 'chest_pain_severity', 
    'abdominal_pain_severity', 'fatigue_level', 'nausea']

bin_edges = np.arange(0, 11, 1)             # [0,1,2,...,10]
bin_edges = np.append(bin_edges, 10.00001)  # Add a tiny bit above 10 to catch only 10
bin_labels = list(range(11))                # Create labels 0-10

for col in generic_symptoms_cols:
    df_v2_binned[col] = pd.cut(
        np.array(df_v2[col]),
        bins=bin_edges,
        labels=bin_labels,          
        include_lowest=True,
        right=False                         # intervals are [a,b) (left-inclusive and right-exclusive)
    )

# Bin the tests columns into bins from TESTS dictionary
def get_test_bin_edges(TESTS):
    """
    Returns a dict mapping test_name -> sorted list of unique bin edges across all diseases.
    """
    bins = {}
    for disease, disease_data in TESTS.items():
        for test_name, intervals in disease_data['tests'].items():
            edges = []
            # Get all min and max for each interval
            for prob, low, high in intervals:
                edges.append((low, high))
            # print(disease, test_name, edges)
            if test_name not in bins:
                bins[test_name] = set()
            bins[test_name].update(edges)
    # Turn sets into sorted lists
    for test_name in bins:
        bins[test_name] = sorted(bins[test_name])
    return bins

test_bin_edges = get_test_bin_edges(TESTS)
print(test_bin_edges)

for test_col, edges in test_bin_edges.items():
    labels = []
    bin_edges = {edges[0][0]}  # Start with the lowest edge

    # Create labels as midpoints of each interval
    for low, high in edges:
        labels.append((low + high) * 0.5)
        if low != high:
            bin_edges.add(high)
        else:
            bin_edges.add(high + 0.00001)
    
    # Some checks
    print(test_col, "-> Edges:", edges)
    print("Labels:", labels)
    print("Edges for the bins:", list(sorted((bin_edges))), "\n")
    
    # Only bin if column exists in dataset
    if test_col in df_v2.columns:
        df_v2_binned[test_col] = pd.cut(
            np.array(df_v2[test_col]),
            bins=list(sorted(bin_edges)),
            labels=labels,
            include_lowest=True,
            right=True
        )

# Check the result
print(df_v2_binned.head())

# Save the binned version of the dataset
df_v2_binned.to_csv("./synthetic_patient_basic_setting_v2_binned.csv", index=False)
