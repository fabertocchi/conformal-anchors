"""Base anchor functions"""
from __future__ import print_function
import numpy as np
import operator
import copy
import sklearn
import collections
import math


def matrix_subset(matrix, n_samples):
    if matrix.shape[0] == 0:
        return matrix
    n_samples = min(matrix.shape[0], n_samples)
    return matrix[np.random.choice(matrix.shape[0], n_samples, replace=False)]


def _accum_constraints(indices, mapping):
    """From predicate indices -> per-feature constraints."""
    # Dictionary to collect constraints per-feature
    by_feat = {}
    for i in indices:
        f, op, v = mapping[i]
        d = by_feat.setdefault(f, {"eq": None, "geq": None, "leq": None})
        if op == "eq":
            if d["eq"] is None:
                d["eq"] = v
        elif op == "geq":
            # Keep the largest geq
            d["geq"] = v if d["geq"] is None or v > d["geq"] else d["geq"]
        elif op == "leq":
            # Keep the smallest leq
            d["leq"] = v if d["leq"] is None or v < d["leq"] else d["leq"]

        # print("Accumulating constraint:", f, op, v, "->", by_feat[f])
    return by_feat

def check_predicate_conflict(mapping, tuple_indices, new_idx):
    """
    Return True if adding mapping[new_idx] conflicts with constraints implied by tuple_indices.
    Handles eq/eq, eq vs geq/leq, and geq vs leq.
    """
    f, op, v = mapping[new_idx]
    #print("Checking conflict for adding:", f, op, v)
    cons = _accum_constraints(tuple_indices, mapping)

    # if current tuple has no constraints for this feature: no conflict
    if f not in cons:
        return False

    # Unpack existing constraints for the feature
    eqv, lo, hi = cons[f]["eq"], cons[f]["geq"], cons[f]["leq"]
    # print("Existing constraints:", f, "eq:", eqv, "geq:", lo, "leq:", hi)

    def isnum(x): return (x is not None) and not (isinstance(x, float) and math.isnan(x))

    if op == "eq":
        # If there is already an equality and it is different: conflict
        if eqv is not None and eqv != v: return True
        # It must satisfy existing geq/leq (conflict if it falls outside)
        if isnum(lo) and isnum(v) and v < lo: return True
        if isnum(hi) and isnum(v) and v > hi: return True

    elif op == "geq":
        # If there is an equality and it's below the new lower bound: conflict
        if isnum(eqv) and isnum(v) and eqv < v: return True
        # If there is upper bound but new lower bound is above it: conflict
        if isnum(hi) and isnum(v) and v > hi: return True

    elif op == "leq":
        # If there is an equality and it's above the new upper bound: conflict
        if isnum(eqv) and isnum(v) and eqv > v: return True
        # If there is lower bound but new upper bound is below it: conflict
        if isnum(lo) and isnum(v) and v < lo: return True

    return False

class AnchorBaseBeam(object):
    def __init__(self):
        pass

    @staticmethod
    def kl_bernoulli(p, q):
        """Computes KL divergence between two Bernoulli distributions with parameters p and q,
        measuring how different they are."""
        p = min(0.9999999999999999, max(0.0000001, p))
        q = min(0.9999999999999999, max(0.0000001, q))
        return (p * np.log(float(p) / q) + (1 - p) *
                np.log(float(1 - p) / (1 - q)))

    @staticmethod
    def dup_bernoulli(p, level):
        """Computes the upper confidence bound on the Bernoulli mean
        performing a single midpoint update."""
        lm = p
        um = min(min(1, p + np.sqrt(level / 2.)), 1)
        qm = (um + lm) / 2.
        # Check whether KL(p, qm) exceeds the target level and moves 'um' or 'lm' accordingly
        if AnchorBaseBeam.kl_bernoulli(p, qm) > level:
            um = qm
        else:
            lm = qm
        return um

    @staticmethod
    def dlow_bernoulli(p, level):
        """Computes the lower confidence bound on the Bernoulli mean
        performing a single midpoint update."""
        um = p
        lm = max(min(1, p - np.sqrt(level / 2.)), 0)
        qm = (um + lm) / 2.
        if AnchorBaseBeam.kl_bernoulli(p, qm) > level:
            lm = qm
        else:
            um = qm
        return lm

    @staticmethod
    def compute_beta(n_features, t, delta):
        """Computes the beta(t, delta) exploration parameter 
        controlling how wide the confidence intervals should be at iteration t"""
        alpha = 1.1
        k = 405.5
        temp = np.log(k * n_features * (t ** alpha) / delta)
        return temp + np.log(temp)

    @staticmethod
    def lucb(sample_fns, initial_stats, epsilon, delta, batch_size, top_n,
             verbose=False, verbose_every=1):
        
        # initial_stats must have n_samples, positive
        n_features = len(sample_fns)
        n_samples = np.array(initial_stats['n_samples'])
        positives = np.array(initial_stats['positives'])
        ub = np.zeros(n_samples.shape)
        lb = np.zeros(n_samples.shape)

        for f in np.where(n_samples == 0)[0]:
            n_samples[f] += 1
            positives[f] += sample_fns[f](1)
            # print("helpppp", sample_fns[f](1))
        if n_features == top_n:
            return range(n_features)
        
        means = positives / n_samples
        t = 1

        def update_bounds(t):
            sorted_means = np.argsort(means)
            # Compute beta exploration parameter controlling confidence interval width
            beta = AnchorBaseBeam.compute_beta(n_features, t, delta)

            # Split candidates into current top_n with largest means (J) and the rest (not_J)
            J = sorted_means[-top_n:]
            not_J = sorted_means[:-top_n]

            # Computes upper bounds for non-top_n arms and lower bounds for top_n arms
            for f in not_J:
                ub[f] = AnchorBaseBeam.dup_bernoulli(means[f], beta /
                                                     n_samples[f])
            for f in J:
                lb[f] = AnchorBaseBeam.dlow_bernoulli(means[f],
                                                      beta / n_samples[f])
            
            # Identify the "most uncertain" arms to sample next
            ut = not_J[np.argmax(ub[not_J])]        # Uncertain top loser (non-top rule with highest upper bound)
            lt = J[np.argmin(lb[J])]                # Uncertain top winner (top rule with lowest lower bound)
            
            return ut, lt
        
        ut, lt = update_bounds(t)
        # Initial gap between confidence intervals
        B = ub[ut] - lb[lt]
        verbose_count = 0
        while B > epsilon:
            # While uncertainty gap is greater than tolerance epsilon, keep sampling
            verbose_count += 1
            if verbose and verbose_count % verbose_every == 0:
                print('Best: %d (mean:%.10f, n: %d, lb:%.4f)' %
                      (lt, means[lt], n_samples[lt], lb[lt]), end=' ')
                print('Worst: %d (mean:%.4f, n: %d, ub:%.4f)' %
                      (ut, means[ut], n_samples[ut], ub[ut]), end=' ')
                print('B = %.2f' % B)
            # Sample both arms to tighten their bounds
            n_samples[ut] += batch_size
            positives[ut] += sample_fns[ut](batch_size)
            means[ut] = positives[ut] / n_samples[ut]

            n_samples[lt] += batch_size
            positives[lt] += sample_fns[lt](batch_size)
            means[lt] = positives[lt] / n_samples[lt]

            t += 1
            # Recompute bounds and uncertainty gap
            ut, lt = update_bounds(t)
            B = ub[ut] - lb[lt]
        
        # When bounds are sufficiently separated, return the top_n arms with largest means
        sorted_means = np.argsort(means)
        if verbose:
            print("means", sorted_means[-top_n:])
        return sorted_means[-top_n:]

    @staticmethod
    def make_tuples(previous_best, state):
        """Defines how new candidate rules are created at each iteration of beam search"""
        
        normalize_tuple = lambda x: tuple(sorted(set(x)))       # Ensures tuples are always ordered and unique
        
        # Retrieve necessary state information
        all_features = range(state['n_features'])               # Indices of all possible predicates
        coverage_data = state['coverage_data']                  # Presampled data used to compute coverage of anchors (without constraints)
        current_idx = state['current_idx']
        data = state['data'][:current_idx]                      # Part of sampled data actually filled so far
        labels = state['labels'][:current_idx]                  # Corresponding labels for filled part of sampled data
        
        # If no previous anchors, create all single-predicate tuples
        if len(previous_best) == 0:
            tuples = [(x, ) for x in all_features][::-1] # reversed order of tuples
            for x in tuples:
                pres = data[:, x[0]].nonzero()[0]                   # Samples satisfying the predicate x
                state['t_idx'][x] = set(pres)                       # Indices of samples satisfying predicate x
                state['t_nsamples'][x] = float(len(pres))           # Number of samples satisfying predicate x
                state['t_positives'][x] = float(labels[pres].sum()) # Number of positive samples satisfying predicate x
                state['t_order'][x].append(x[0])                    # Order in which predicates were added
                state['t_coverage_idx'][x] = set(                   # Indices in 'coverage_data' where predicate holds
                    coverage_data[:, x[0]].nonzero()[0])
                state['t_coverage'][x] = (                          # Coverage = len(samples with that predicate active) / total samples
                    float(len(state['t_coverage_idx'][x])) /
                    coverage_data.shape[0])
                # print("x", x)
                # print("pres", pres)
                # print("t_idx", state['t_idx'][x])
                # print("t_nsamples", state['t_nsamples'][x])
                # print("t_positives", state['t_positives'][x])
                # print("t_order", state['t_order'][x])
                # print('coverage2: ', coverage_data[:, x[0]], coverage_data[:, x[0]].shape)
                # print("len t_coverage_idx", len(state['t_coverage_idx'][x]))
                # print("t_coverage", state['t_coverage'][x])
            # print(state['t_idx'], state['t_nsamples'], state['t_positives'], state['t_order'], state['t_coverage'])
            # print("initial tuples:", tuples)
            return tuples
        
        
        # Later iterations: expand previous best anchors by adding one new predicate
        new_tuples = set()
        # print("previous_best", previous_best)
        for f in all_features:
            for t in previous_best:
                if state.get('enable_conflict_check', True) and state.get('mapping') is not None:
                
                # 1) against the current tuple t
                    if check_predicate_conflict(state['mapping'], t, f):
                        # print("Conflict detected when adding", f, "to", t)
                        continue
                    # 2) against the current best (store it in state before this call; see below)
                    best_t = state.get('best_tuple_indices')
                    if best_t and check_predicate_conflict(state['mapping'], best_t, f):
                        # print("Conflict detected when adding", f, "to best", best_t)
                        continue
                # print("t", t)
                # Add feature f to existing tuple t
                new_t = normalize_tuple(t + (f, ))
                # print("Trying to create new tuple", new_t)
                # print("before new_t", new_t)
                
                # Skip adding f if it is already present in t
                if len(new_t) != len(t) + 1:
                    #print("skipped", new_t, "already present in", t)
                    continue
                
                # If new tuple not already created 
                if new_t not in new_tuples:
                    # print("adding", new_t)
                    new_tuples.add(new_t)
                    # print("new_tuples", new_tuples)
                    # Store the feature order of this new rule inheriting from parent t
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    # Compute coverage indices of new_t as the intersection of parent's indices coverage and of the new predicate f
                    state['t_coverage_idx'][new_t] = (
                        state['t_coverage_idx'][t].intersection(state['t_coverage_idx'][(f,)])
                    )
                    # Compute coverage of the new rule
                    state['t_coverage'][new_t] = (
                        float(len(state['t_coverage_idx'][new_t])) /
                        coverage_data.shape[0])
                    
                    t_idx = np.array(list(state['t_idx'][t]))           # Indices of samples satisfying parent rule t
                    t_data = state['data'][t_idx]                       # Corresponding row samples
                    present = np.where(t_data[:, f] == 1)[0]            # Among them, where new predicate is active
                    state['t_idx'][new_t] = set(t_idx[present])         # Indices satisfying new rule
                    idx_list = list(state['t_idx'][new_t])              # List of indices satisfying new rule
                    state['t_nsamples'][new_t] = float(len(idx_list))   # Number of samples satisfying new rule
                    state['t_positives'][new_t] = np.sum(               # Number of positives among them
                        state['labels'][idx_list])
        # print("output", sorted(list(new_tuples)), len(list(new_tuples)))
        return list(new_tuples)

    @staticmethod
    def get_sample_fns(sample_fn, tuples, state):
        # Each sample fn returns number of positives
        sample_fns = []

        def complete_sample_fn(t, n):
            """Draw n new perturbations conditioned on rule 't'
            Calls provided 'sample_fn', conditioning on rule 't', to draw n new samples"""
            raw_data, data, labels = sample_fn(list(t), n)
            
            current_idx = state['current_idx']
            idxs = range(current_idx, current_idx + n)          # Where in the preallocated buffer these new n rows will be written

            if '<U' in str(raw_data.dtype):
                # String types: make sure both string types are of maximum length 
                # to avoid string truncation. E.g., '<U308', '<U290' -> '<U308'
                max_dtype = max(str(state['raw_data'].dtype), str(raw_data.dtype))
                state['raw_data'] = state['raw_data'].astype(max_dtype)
                raw_data = raw_data.astype(max_dtype)

            # Add indices of these new rows (they are built to satisfy rule t)
            state['t_idx'][t].update(idxs)
            # Add number of generated samples
            state['t_nsamples'][t] += n
            # Add number of positive samples in generated samples
            state['t_positives'][t] += labels.sum()
            # Write new samples into preallocated buffers
            state['data'][idxs] = data
            state['raw_data'][idxs] = raw_data
            state['labels'][idxs] = labels
            # Slide current index forward by n
            state['current_idx'] += n

            # Autogrow the buffers when we're close to full capacity
            if state['current_idx'] >= state['data'].shape[0] - max(1000, n):
                prealloc_size = state['prealloc_size']
                current_idx = data.shape[0]
                state['data'] = np.vstack(
                    (state['data'],
                     np.zeros((prealloc_size, data.shape[1]), data.dtype)))
                state['raw_data'] = np.vstack(
                    (state['raw_data'],
                     np.zeros((prealloc_size, raw_data.shape[1]),
                              raw_data.dtype)))
                state['labels'] = np.hstack(
                    (state['labels'],
                     np.zeros(prealloc_size, labels.dtype)))
            
            # Return number of positives in this batch
            return labels.sum()
        
        # Build one lambda function per tuple t
        for t in tuples:
            sample_fns.append(lambda n, t=t: complete_sample_fn(t, n))

        return sample_fns


    @staticmethod
    def get_initial_statistics(tuples, state):
        stats = {
            'n_samples': [],
            'positives': []
        }
        for t in tuples:
            stats['n_samples'].append(state['t_nsamples'][t])
            stats['positives'].append(state['t_positives'][t])
        return stats

    @staticmethod
    def get_anchor_from_tuple(t, state):
        """Takes the tuple of predicates indices that defines an anchor 
        and reconstructs a full, human-readable anchor object with all its statistics"""
        # TODO: This is wrong, some of the intermediate anchors may not exist. -> theoretical concern
        # (Warns that some intermediate subsets may not exist if they weren’t stored during beam search)
        
        # Initialize output dictionary
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        anchor['num_preds'] = state['data'].shape[0]
        normalize_tuple = lambda x: tuple(sorted(set(x)))     # Ensures tuples are always ordered and unique
        current_t = tuple()
        for f in state['t_order'][t]:
            # Iteratively examine each intermediate rule of the anchor
            # print("f", f)
            current_t = normalize_tuple(current_t + (f,))
            if current_t not in state['t_nsamples']:
                print("Missing intermediate anchor:", current_t)

            # Compute empirical precision
            mean = (state['t_positives'][current_t] /
                    state['t_nsamples'][current_t])
            anchor['feature'].append(f)
            anchor['mean'].append(mean)
            anchor['precision'].append(mean)
            anchor['coverage'].append(state['t_coverage'][current_t])
            # Collect representative examples
            raw_idx = list(state['t_idx'][current_t])                   # Indices of all samples satisfying current rule
            raw_data = state['raw_data'][raw_idx]                       # Corresponding raw data samples
            # Examples correctly predicted
            covered_true = (
                state['raw_data'][raw_idx][state['labels'][raw_idx] == 1])
            # Examples incorrectly predicted
            covered_false = (
                state['raw_data'][raw_idx][state['labels'][raw_idx] == 0])
            exs = {}
            # Pick up to 10 random rows for illustration
            exs['covered'] = matrix_subset(raw_data, 10)
            exs['covered_true'] = matrix_subset(covered_true, 10)
            exs['covered_false'] = matrix_subset(covered_false, 10)
            exs['uncovered_true'] = np.array([])
            exs['uncovered_false'] = np.array([])
            anchor['examples'].append(exs)            

        return anchor

    @staticmethod
    def anchor_beam(sample_fn, delta=0.05, epsilon=0.1, batch_size=10,
                    min_shared_samples=0, desired_confidence=1, beam_size=10,
                    verbose=False, epsilon_stop=0.05, min_samples_start=0,
                    max_anchor_size=None, verbose_every=1,
                    stop_on_first=False, coverage_samples=10000, mapping=None, enable_conflict_check=False):
        """Search for a high-precision anchor using the beam search strategy."""
       
        # Placeholders for the final anchor statistics
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        # Draw a large, unconstrained batch to estimate coverage probabilities later
        raw_coverage_data, coverage_data, raw_labels = sample_fn([], coverage_samples, compute_labels=False)
        
        # We sample a small initial batch of perturbed instances (min_samples_start, typically 1) without any constraints (empty rule)
        # and check whether the model keeps the same prediction f(x) under these perturbations.
        # 'labels' contains 1 if f(z) == f(x), 0 otherwise.
        raw_data, data, labels = sample_fn([], max(1, min_samples_start))
        mean = labels.mean()                    # Initial estimate of precision for empty anchor           
        beta = np.log(1. / delta)               # Confidence parameter for the statistical bounds
        # Compute the lower confidence bound (LCB) on the true precision using the KL-based inequality.
        # This gives a conservative estimate: with high probability (1−δ), the true precision ≥ lb.
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
        # print(f"Initial mean precision: {mean}, LCB: {lb}, beta: {beta}, n_samples: {data.shape[0]}")
        
        # Sanity check: If the model prediction f(x) is already very stable under random perturbations (i.e., almost always the same),
        # the empty rule itself might qualify as an anchor.
        # The loop continues sampling more data ONLY IF:
        #  - The empirical mean precision so far exceeds the desired confidence τ,
        #    meaning it *looks* like a valid anchor,
        #  - BUT the lower bound is still below τ − ε, meaning we don't have enough evidence yet.
        while mean > desired_confidence and lb < desired_confidence - epsilon:
            print("Sampling more data to validate empty rule as anchor...")
            # Draw a new batch of perturbed samples (no fixed predicates)
            nraw_data, ndata, nlabels = sample_fn([], batch_size)
            # Append new samples to existing arrays
            data = np.vstack((data, ndata))
            raw_data = np.vstack((raw_data, nraw_data))
            labels = np.hstack((labels, nlabels))
            # Update empirical mean and recompute lower confidence bound
            mean = labels.mean()
            lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
        # This loop adaptively collects more samples to tighten the confidence interval
        # until we can statistically confirm (or reject) that the empty rule satisfies the precision requirement.
        
        # If we're confident that the empty rule is already a valid anchor, return it.
        if lb > desired_confidence:
            anchor['num_preds'] = data.shape[0]
            anchor['all_precision'] = mean
            return anchor
        
        # If not, preallocate space for storing sampled data during the beam search
        prealloc_size = batch_size * 10000                                      # How many empty rows of extra space to preallocate
        current_idx = data.shape[0]                                             # Index of next free position in the 'data' array

        # Append additional block of zeros (of size (prealloc_size, data.shape[1])) to existing 'data' array
        data = np.vstack((data, np.zeros((prealloc_size, data.shape[1]), data.dtype)))
        # Do the same for 'raw_data' and 'labels'
        #print("raw_data shape:", raw_data.shape)
        raw_data = np.vstack((raw_data, np.zeros((prealloc_size, raw_data.shape[1]), raw_data.dtype)))
        #print("raw_data shape:", raw_data.shape)
        labels = np.hstack((labels, np.zeros(prealloc_size, labels.dtype)))
        # Number of predicates
        n_features = data.shape[1]
        
        # Initialize 'state' dictionary to keep track of statistics during beam search
        state = {'t_idx': collections.defaultdict(lambda: set()),               # Indices of samples satisfying each anchor tuple
                 't_nsamples': collections.defaultdict(lambda: 0.),             # Number of samples satisfying each anchor tuple
                 't_positives': collections.defaultdict(lambda: 0.),            # Number of positive (f(z)=f(x)) samples satisfying each anchor tuple
                 'data': data,
                 'prealloc_size': prealloc_size,
                 'raw_data': raw_data,
                 'labels': labels,
                 'current_idx': current_idx,
                 'n_features': n_features,
                 't_coverage_idx': collections.defaultdict(lambda: set()),      # Coverage indices for each anchor tuple
                 't_coverage': collections.defaultdict(lambda: 0.),             # Coverage values for each anchor tuple
                 'coverage_data': coverage_data,
                 't_order': collections.defaultdict(lambda: list()),
                 'mapping': mapping,
                 #'enable_conflict_check': bool(enable_conflict_check)
                 }
        valid_anchors_list = []
        # Beam search loop: iteratively grow the current best rules by adding one predicate at a time
        current_size = 1                                            # Current size of the anchors being considered    
        best_of_size = {0: []}                                      # Best anchors found of each size
        best_coverage = -1                                          # Best coverage observed so far 
        best_tuple = ()                                             # Tuple representing the best anchor found so far
        # t = 1
        if max_anchor_size is None:
            # If not specified, set maximum anchor size to number of predicates
            max_anchor_size = n_features

        # Expand anchors by one predicate at a time, up to max_anchor_size
        while current_size <= max_anchor_size:
            # print("----------------")
            # print("Current size:", current_size)
            # Generate all candidate tuples of length 'current_size' by extending the best anchors of size 'current_size - 1'
            tuples = AnchorBaseBeam.make_tuples(
                best_of_size[current_size - 1], state)
            # print("All tuples of size", current_size, ":", tuples, "len tuples:", len(tuples))
            # print("tuples before filtering:", tuples)
            # Filter out tuples with coverage less than the best found so far
    
            tuples = [x for x in tuples
                      if state['t_coverage'][x] > best_coverage]
            # print("Filtered tuples:", tuples)
            if len(tuples) == 0:
                print("no valid tuples found")
                break
            # Prepare sampling functions and initial statistics for the current set of candidate tuples
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples,
                                                       state)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                                  state)
            # Use KL-LUCB to efficiently identify top-B candidates by precision
            # print(min(beam_size, len(tuples)), "tuples will be selected by KL-LUCB")
            chosen_tuples = AnchorBaseBeam.lucb(
                sample_fns, initial_stats, epsilon, delta, batch_size,
                min(beam_size, len(tuples)),
                verbose=verbose, verbose_every=verbose_every)
            if verbose:
                print("chosen tuples:", chosen_tuples, "len chosen tuples:", len(chosen_tuples))
            # Keep the B-best rules of the current size for the next iteration
            best_of_size[current_size] = [tuples[x] for x in chosen_tuples]
            # print(f"Best of size {current_size}:", best_of_size[current_size])
            if verbose:
                print('Best of size ', current_size, ':')
            
            stop_this = False
            for i, t in zip(chosen_tuples, best_of_size[current_size]):
                #print("Evaluating tuple:", i, t)
                # I can choose at most (beam_size - 1) tuples at each step,
                # and there are at most n_feature steps
                # Compute confidence bounds for the candidate rule
                beta = np.log(1. / (delta / (1 + (beam_size - 1) * n_features)))
                mean = state['t_positives'][t] / state['t_nsamples'][t]         # Empirical precision estimate
                lb = AnchorBaseBeam.dlow_bernoulli(
                    mean, beta / state['t_nsamples'][t])
                ub = AnchorBaseBeam.dup_bernoulli(
                    mean, beta / state['t_nsamples'][t])
                coverage = state['t_coverage'][t]

                if verbose:
                    print(
                        f"[SIZE {current_size}] candidate={t} | "
                        f"cov={coverage:.6f} | mean={mean:.4f} | lb={lb:.4f} | ub={ub:.4f} | "
                        f"n={int(state['t_nsamples'][t])}"
                    )


                if verbose:
                    print("chosen tuple:", i, "precision:", mean, "lb:", lb, "ub:", ub)
                # Keep sampling until the confidence interval is sufficiently tight
                while ((mean >= desired_confidence and lb < desired_confidence - epsilon_stop) or
                       (mean < desired_confidence and ub >= desired_confidence + epsilon_stop)):
                    # sample_fns[i] updates t_nsamples and t_positives in state in-place
                    sample_fns[i](batch_size)
                    mean = state['t_positives'][t] / state['t_nsamples'][t]
                    lb = AnchorBaseBeam.dlow_bernoulli(
                        mean, beta / state['t_nsamples'][t])
                    ub = AnchorBaseBeam.dup_bernoulli(
                        mean, beta / state['t_nsamples'][t])
                    
                if verbose:
                    print(
                        f"[SIZE {current_size}] final  ={t} | "
                        f"cov={coverage:.6f} | mean={mean:.4f} | lb={lb:.4f} | ub={ub:.4f} | "
                        f"n={int(state['t_nsamples'][t])}"
                    )

                
                # print('%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' % (t, mean, lb, ub, coverage, state['t_nsamples'][t]))
                # If precision is confidently above the threshold => valid anchor
                if mean >= desired_confidence and lb > desired_confidence - epsilon_stop:
                    if coverage > 0:
                        #print("Found valid anchor:", t, "precision:", mean, "lb:", lb, "coverage:", coverage)
                        valid_anchors_list.append(t)
                    # print('Found eligible anchor ', t, 'Coverage:',
                              # coverage, 'Is best?', coverage > best_coverage)
                    # Choose the anchor with highest coverage among valid ones
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_tuple = t
                        if verbose:
                            print("Best tuple with coverage", best_coverage, ":", best_tuple)
                        # Stop early if desired coverage reached or user requests early stop
                        # if best_coverage == 1 or stop_on_first:
                        #     stop_this = True
            if stop_this:
                break
            current_size += 1
            # print("Current size increased to", current_size)
        # If no anchor satisfied the confidence condition
        if best_tuple == ():
            # Could not find an anchor, will now choose the highest precision
            # amongst the top K from every round
            #if verbose:
            print('Could not find an anchor, now doing best of each size')
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])
            # tuples now contains all best_of_size from each round
            # Prepare sampling functions and initial statistics for these tuples
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples,
                                                       state)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                                  state)
            # Identify most precise anchor among these candidates
            chosen_tuples = AnchorBaseBeam.lucb(
                sample_fns, initial_stats, epsilon, delta, batch_size,
                1, verbose=verbose)
            best_tuple = tuples[chosen_tuples[0]]
            if verbose:
                print("Best tuple:", best_tuple)

    
        print("---> valid anchors found:", len(valid_anchors_list), valid_anchors_list)
        # Create a list of anchor dicts
        valid_anchors_dict = [AnchorBaseBeam.get_anchor_from_tuple(t, state) for t in valid_anchors_list]

        if len(valid_anchors_list) > 0:
            covered_union = set()
            for t in valid_anchors_list:
                covered_union |= state['t_coverage_idx'][t]  # union of indices: adds all indices without duplicates
            cumulative_coverage = float(len(covered_union)) / coverage_data.shape[0]
        else:
            cumulative_coverage = 0.0

        best_anchor = AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state)
        best_anchor['cumulative_coverage'] = cumulative_coverage


        # print("Cumulative coverage of all valid anchors:", cumulative_coverage)

        # Return the final anchor as a readable structure
        return best_anchor, valid_anchors_dict
