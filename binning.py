import pandas as pd
import numpy as np
from data.tests_v2 import TESTS
# from data.tests_v1 import TESTS

# Bin the tests columns into bins from TESTS dictionary
def get_test_bin_edges(TESTS):
    """
    Build a dictionary mapping each diagnostic test to the full set of 
    (low, high) intervals across all diseases.

    TESTS structure:
        TESTS[disease]['tests'][test_name] = [(prob, low, high), ...]
    
    For each test, the function collects all intervals from all diseases,
    removes duplicates, and returns a sorted list of (low, high) tuples.

    Returns
    -------
    bins : dict
        test_name -> sorted list of unique (low, high) interval tuples
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

    # Convert sets into sorted lists
    for test_name in bins:
        bins[test_name] = sorted(bins[test_name])

    return bins

def get_test_bin_labels(TESTS):
    """
    Returns:
      dict: test_name -> sorted list of unique bin labels (midpoints)
    """
    test_bin_edges = get_test_bin_edges(TESTS)
    bin_labels = {}

    for test_name, edges in test_bin_edges.items():
        labels = [(low + high) * 0.5 for (low, high) in edges]
        bin_labels[test_name] = sorted(set(labels))

    return bin_labels

TEST_BIN_LABELS = get_test_bin_labels(TESTS)

def bin_dataset(df, TESTS, generic_symptoms_cols=None, verbose=False):
    """
    Create a binned version of the dataset according to:
        - fixed 0–10 bins for generic symptoms
        - dynamic bins (interval midpoints) for diagnostic tests
          derived from the TESTS dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to be binned.

    TESTS : dict
        The dictionary specifying intervals for each diagnostic test
        (used to compute bin boundaries).

    generic_symptoms_cols : list of str, optional
        The columns that should be binned into 11 bins (0–10). If None,
        a default list is used.

    verbose : bool
        If True, print bin edges and labels for each test.

    Returns
    -------
    df_binned : pd.DataFrame
        Copy of df with binned numeric values replaced by category labels.
    """
    df_binned = df.copy()

    # Bin generic symptoms into 0-10
    if generic_symptoms_cols is None:
        generic_symptoms_cols = ['fever_severity', 'cough_severity', 'chest_pain_severity', 
            'abdominal_pain_severity', 'fatigue_level', 'nausea']
        
    bin_edges = np.arange(0, 11, 1)             # [0,1,2,...,10]
    bin_edges = np.append(bin_edges, 10.00001)  # Add a tiny bit above 10 to catch only 10
    bin_labels = list(range(11))                # Create labels 0-10

    for col in generic_symptoms_cols:
        if col in df_binned.columns:
            df_binned[col] = pd.cut(
                np.array(df_binned[col]),
                bins=bin_edges,
                labels=bin_labels,          
                include_lowest=True,
                right=False                         # intervals are [a,b) (left-inclusive and right-exclusive)
            )
    
    # Bin diagnostic tests using TESTS
    test_bin_edges = get_test_bin_edges(TESTS)

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

        if verbose:
            print(test_col, "-> Edges:", edges)
            print("Labels:", labels)
            print("Edges for the bins:", list(sorted((bin_edges))), "\n")
        
        # Only bin if column exists in dataset
        if test_col in df_binned.columns:
            df_binned[test_col] = pd.cut(
                np.array(df_binned[test_col]),
                bins=list(sorted(bin_edges)),
                labels=labels,
                include_lowest=True,
                right=True
            )
    
    return df_binned

if __name__ == "__main__":
    # Upload the two versions of the dataset
    df_v1 = pd.read_csv("generated_datasets/synthetic_patient_basic_setting_v1.csv")
    df_v2 = pd.read_csv("generated_datasets/synthetic_patient_basic_setting_v2.csv")

    # Define the names of generic symptoms columns
    generic_symptoms_cols = ['fever_severity', 'cough_severity', 'chest_pain_severity', 
    'abdominal_pain_severity', 'fatigue_level', 'nausea']

    # Bin the dataset
    df_v2_binned = bin_dataset(df_v2, TESTS, generic_symptoms_cols=generic_symptoms_cols, verbose=True)
    # Check the result
    print(df_v2_binned.head())
    # Print the unique values for each binned test column
    print(TEST_BIN_LABELS)

    # Save the binned version of the dataset
    df_v2_binned.to_csv("generated_datasets/synthetic_patient_basic_setting_v2_binned.csv", index=False)
    
