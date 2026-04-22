from . import anchor_base
from . import anchor_explanation
from . import utils
import lime
import lime.lime_tabular
import collections
import sklearn
import numpy as np
import os
import copy
import string
from io import open
import json

def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

class AnchorTabularExplainer(object):
    """
    Args:
        class_names: list of strings
        feature_names: list of strings
        train_data: used to sample (bootstrap)
        categorical_names: map from integer to list of strings, names for each
            value of the categorical features. Every feature that is not in
            this map will be considered as ordinal or continuous, and thus discretized.
    """
    def __init__(self, class_names, feature_names, train_data,
                 categorical_names={}, discretizer=None, encoder_fn=None):
        
        # Initialize min/max dictionaries which will hold the min and max values for each feature
        self.min = {}
        self.max = {}

        # Store the encoder function, defaulting to the identity when no preprocessing is needed
        self.encoder_fn = encoder_fn if encoder_fn is not None else (lambda x: x)
        
        # Record metadata about the training dataset
        self.feature_names = feature_names
        self.train = train_data
        self.class_names = class_names
        self.categorical_names = copy.deepcopy(categorical_names)
        # Extracts list of categorical feature indices
        self.categorical_features = sorted(categorical_names.keys()) if categorical_names else []

        # If no discretizer is supplied, use identity function and treat stored training data as already discretized
        if discretizer is None:
            # No discretization case: identity function
            self.disc = collections.namedtuple('NoDisc', ['discretize'])(lambda x: x)
            self.d_train = self.train  # Already binned data

            # If categorical names were not provided, infer them enumerating unique values in each column of training dataset
            if not self.categorical_names:
                self.categorical_names = {
                    i: [v for v in sorted(np.unique(train_data[:, i]))]
                    for i in range(train_data.shape[1])
                }
                print("Categorical names:", self.categorical_names)
        
        # If a discretizer is provided, initialize LIME discretizer accordingly
        else:
            if discretizer == 'quartile':
                self.disc = lime.lime_tabular.QuartileDiscretizer(
                    train_data, self.categorical_features, self.feature_names)
            elif discretizer == 'decile':
                self.disc = lime.lime_tabular.DecileDiscretizer(
                    train_data, self.categorical_features, self.feature_names)
            else:
                raise ValueError('Discretizer must be quartile, decile, or None')
            self.d_train = self.disc.discretize(self.train)
            self.categorical_names.update(self.disc.names)

        # Identify ordinal features as those not listed in categorical_features
        self.ordinal_features = [x for x in range(len(feature_names)) if x not in self.categorical_features]
        # Extend categorical features to include ordinal features as well
        self.categorical_features += self.ordinal_features
        # print("Categorical features:", self.categorical_features, "Ordinal features:", self.ordinal_features)

        # Min/max per sampling
        for f in range(self.train.shape[1]):            # Iterate over each column of the dataset (feature)
            # Record min and max observed values for each feature
            self.min[f] = np.min(self.train[:, f])
            self.max[f] = np.max(self.train[:, f])

    def sample_from_train(self, conditions_eq, conditions_neq, conditions_geq,
                          conditions_leq, num_samples):
        """
        Samples rows from the training set satisfying the given feature conditions.
        """
        train = self.train          # train_data
        d_train = self.d_train      # discretized train_data (which in case of Discretizer=None is the same as train_data)
        #print("train shape:", train.shape[0])
        # print("range", list(range(train.shape[0])))
        #print("num_samples:", num_samples)

        # Sample candidate indices with replacement: pick `num_samples` row indices from the training set
        idx = np.random.choice(range(train.shape[0]), num_samples, replace=True)        
        # print(idx)
        sample = train[idx]
        #print("Sampled rows before applying conditions:", sample)
        d_sample = d_train[idx]

        # Apply the conditions to the sampled rows overwriting them
        # Force equality constraints
        for f in conditions_eq:
            sample[:, f] = np.repeat(conditions_eq[f], num_samples)     # `conditions_eq` is a dict {feature_index: value}
        #print("eq", conditions_eq, "geq", conditions_geq, "leq", conditions_leq)
        # Adjust lower-bound (>) constraints
        for f in conditions_geq:
            # Identify which sampled rows violate the '> condition' for feature f
            violating = d_sample[:, f] <= conditions_geq[f]             # Boolean mask
            #print(violating, len(violating))
            # If same feature has also an upper-bound (<=) constraint, 
            # combine the masks so that `violating` becomes true for any row outside the closed interval `[conditions_geq[f], conditions_leq[f]]`
            if f in conditions_leq:
                violating = (violating + (d_sample[:, f] > conditions_leq[f])).astype(bool)
                #print("violating combined with leq", violating)
            # If no rows violate the condition, skip to next feature
            if violating.sum() == 0:
                continue

            # Search the training set for candidate replacement values that satisfy the lower bound.
            # (Prefer resampling from training rows that already satisfy the bounds)
            options = d_train[:, f] > conditions_geq[f]
            # When upper bound exists, tighten the search to rows lying inside the admissible interval
            if f in conditions_leq:
                options = options * (d_train[:, f] <= conditions_leq[f])

            # If no training rows satisfy the constraints, fall back to uniform sampling within the valid range
            if options.sum() == 0:
                # Determine admissible category values
                valid_vals = np.array(self.categorical_names[f])
                if f in conditions_leq:
                    valid_vals = valid_vals[(valid_vals > conditions_geq[f]) &
                                            (valid_vals <= conditions_leq[f])]
                else:
                    valid_vals = valid_vals[valid_vals > conditions_geq[f]]

                # If still empty, fallback to all possible values (rare)
                if len(valid_vals) == 0:
                    valid_vals = np.array(self.categorical_names[f])
                # print("f", f, "conditions_geq", conditions_geq[f], "conditions_leq", conditions_leq.get(f, None))
                # print("options sum 0", options)
                # print("to_rep with uniform")
                # min_ = conditions_geq.get(f, self.min[f])
                # max_ = conditions_leq.get(f, self.max[f])
                #print("summmmmm", violating.sum())
                #print("min_", min_, "max_", max_)
                # Synthesise replacement values uniformly at random within the allowed range
                # to_rep = np.random.uniform(min_, max_, violating.sum())
                to_rep = np.random.choice(valid_vals, violating.sum(), replace=True)
                # print(to_rep)
            # Otherwise, reuse actual feature values from compliant training rows
            else:
                to_rep = np.random.choice(train[options, f], violating.sum(),
                                          replace=True)
                #print("to_rep with choice")
            # Overwrite the violating sampled rows with the selected replacement values
            sample[violating, f] = to_rep

        # Adjust upper-bound (<=) constraints when no lower bound was present
        for f in conditions_leq:
            if f in conditions_geq:     # Already handled in previous loop
                continue
            # Identify which sampled rows violate the '<= condition' for feature f
            violating = d_sample[:, f] > conditions_leq[f]
            # If no rows violate the condition, skip to next feature
            if violating.sum() == 0:
                continue
            # Search the training set for candidate replacement values that satisfy the upper bound.
            # (Prefer resampling from training rows that already satisfy the bound)
            options = d_train[:, f] <= conditions_leq[f]
            # When no training rows satisfy the constraints, fall back to uniform sampling within the valid range
            if options.sum() == 0:
                valid_vals = np.array(self.categorical_names[f])
                valid_vals = valid_vals[valid_vals <= conditions_leq[f]]

                if len(valid_vals) == 0:
                    valid_vals = np.array(self.categorical_names[f])
                #print("options sum 0", options)
                # print("to_rep with uniform")
                # min_ = conditions_geq.get(f, self.min[f])
                # max_ = conditions_leq.get(f, self.max[f])
                # to_rep = np.random.uniform(min_, max_, violating.sum())
                to_rep = np.random.choice(valid_vals, violating.sum(), replace=True)
                # print(to_rep)
            # Otherwise, reuse actual feature values from compliant training rows
            else:
                # print("to_rep with choice")
                to_rep = np.random.choice(train[options, f], violating.sum(),
                                          replace=True)
            
            # Overwrite the violating sampled rows with the selected replacement values
            sample[violating, f] = to_rep

        # After applying all conditions, return the modified sampled rows
        return sample


    def transform_to_examples(self, examples, features_in_anchor=[],
                              predicted_label=None):
        """Converts raw example matrices into structured (name, value, weight) format for explanation output."""
        ret_obj = []

        # Return empty list if no examples are provided (rows to format)
        if len(examples) == 0:
            return ret_obj
        
        # Determine weights for each feature based on whether it is included in the anchor:
        # anchor features marked with predicted label, non-anchor features marked with -1
        weights = [int(predicted_label) if x in features_in_anchor else -1
                   for x in range(examples.shape[1])]
        # Discretize the examples
        examples = self.disc.discretize(examples)  # Will do nothing if discretizer=None

        # Convert each row
        for ex in examples:
            # If feature is categorical/ordinal, map discretized integer back to human-readable category name
            # Otherwise, keep continuous value as is.
            values = [self.categorical_names[i][int(ex[i])]
                      if i in self.categorical_features
                      else ex[i] for i in range(ex.shape[0])]
            # Zip the feature names, resolved values, and weights together into a list of triplets and append it to the result.
            ret_obj.append(list(zip(self.feature_names, values, weights)))
        
        return ret_obj

    def to_explanation_map(self, exp):
        """Convert the Anchor search result `exp` into the front-end data schema."""

        # Pull explained instance
        instance = exp['instance']
        # Pull predicted label
        predicted_label = exp['prediction']
        # Initialize one-hot probability vector which marks the predicted class
        predict_proba = np.zeros(len(self.class_names))
        predict_proba[predicted_label] = 1

        # Process supporting examples
        examples_obj = []
        for i, temp in enumerate(exp['examples'], start=1):
            # Features up to the i-th rule are considered active in this panel.
            features_in_anchor = set(exp['feature'][:i])

            examples_obj.append({
                'coveredFalse': self.transform_to_examples(
                    temp['covered_false'], features_in_anchor, predicted_label),
                'coveredTrue': self.transform_to_examples(
                    temp['covered_true'], features_in_anchor, predicted_label),
                'uncoveredTrue': self.transform_to_examples(
                    temp['uncovered_true'], features_in_anchor, predicted_label),
                'uncoveredFalse': self.transform_to_examples(
                    temp['uncovered_false'], features_in_anchor, predicted_label),
                'covered': self.transform_to_examples(
                    temp['covered'], features_in_anchor, predicted_label)
            })

        # Package the anchor names, per-step precision estimates, coverage figures, and the formatted example sets
        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': examples_obj,
                       'onlyShowActive': False}
        
        weights = [-1 for x in range(instance.shape[0])]
        # Discretize original instance
        instance = self.disc.discretize(instance.reshape(1, -1))[0]  # Identity if no discretizer
        # Translate categorical indices back into names
        values = [self.categorical_names[i][int(instance[i])]
                  if i in self.categorical_features
                  else instance[i] for i in range(instance.shape[0])]
        # Package the result as `(feature name, value, weight)` tuples with neutral weights (`-1`)
        raw_data = list(zip(self.feature_names, values, weights))
        
        # Group everything into a JSON-ready dictionary
        ret = {
            'explanation': explanation,
            'rawData': raw_data,
            'predictProba': list(predict_proba),
            'labelNames': list(map(str, self.class_names)),
            'rawDataType': 'tabular',
            'explanationType': 'anchor',
            'trueClass': False
        }

        return ret

    def as_html(self, exp, **kwargs):
        """Produce a self-contained HTML page that renders the anchor explanation."""
        
        # Convert the explanation into the front-end data schema
        exp_map = self.to_explanation_map(exp)

        # Helper to serialise Python objects as JSON literals inside the HTML string.
        def jsonize(x): return json.dumps(x)

        # Read the JavaScript visualisation bundle that knows how to render anchors.
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()
        # Start building the HTML document with the bundle inlined in a <script> tag.
        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "tabular", {explanation_type});
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(exp_map['labelNames']),
                            predict_proba=jsonize(exp_map['predictProba']),
                            true_class=jsonize(exp_map['trueClass']),
                            explanation=jsonize(exp_map['explanation']),
                            raw_data=jsonize(exp_map['rawData']),
                            explanation_type=jsonize(exp_map['explanationType']))
        out += u'</body></html>'
        
        return out


    def get_sample_fn(self, data_row, classifier, 
                      mode="standard",              # "standard" or "conformal"
                      query_label=None, qhat=None,  # used only if mode="conformal"
                      desired_label=None, 
                      predicate_mode="original",    # "original" or "mean_instances" or "medoid"
                      mean_instances=None,          # used only if predicate_mode="mean_instances"  
                      verbose=True):
        """Prepares the sampling function for generating perturbed samples around a reference example
        and evaluates how often a trained model keeps the same prediction under those perturbations."""
        
        if mode == "conformal":
            # Use probability outputs for conformal prediction sets
            predict_fn = lambda x: classifier.predict_proba(self.encoder_fn(x))
            # In conformal mode, the target class is fixed to 'query_label'
            if query_label is None or qhat is None:
                raise ValueError("Both `query_label` and `qhat` must be provided in conformal mode.")
            true_label = int(query_label)
        else:
            # Standard anchors use hard predictions
            predict_fn = lambda x: classifier.predict(self.encoder_fn(x))
            # Standard mode: preserve model’s prediction or user-specified label
            true_label = desired_label
            if true_label is None:
                preds = predict_fn(data_row.reshape(1, -1))
                true_label = int(preds[0])
        
        # if verbose:
        #     print("True label is:", true_label)

        if predicate_mode == "mean_instances":
            if mean_instances is None:
                raise ValueError("`mean_instances` must be provided in mean_instances mode.")
            instances_for_predicates = mean_instances
            if verbose:
                print("Using mean_instances mode with", len(instances_for_predicates), "instances.")

        elif predicate_mode == "medoid":
            instances_for_predicates = [data_row]
            if verbose:
                print("Using medoid mode.")

        elif predicate_mode == "original":
            # Standard mode: predicates built from the single explained instance
            instances_for_predicates = [data_row]
            if verbose:
                print("Using original mode (single instance).")

        else:
            raise ValueError(f"Unknown predicate_mode: {predicate_mode}")

                
        # Discretize the reference row and build candidate predicates (`mapping`)
        mapping = {}    # will hold a dictionary of candidate predicates: idx -> (feature_index, operator, value)
        
        # data_row_disc = self.disc.discretize(data_row.reshape(1, -1))[0]

        # Explore logical conditions over feature values, enumerating every allowable condition for categorical and ordinal features
        # * Ordinal features produce '<=' and '>=' conditions for each possible threshold value
        # * Nominal categorical features produce '==' conditions for each possible category value
        for base_instance in instances_for_predicates:
            base_instance_disc = self.disc.discretize(base_instance.reshape(1, -1))[0]

            # Each predicate is stored in `mapping` using an integer key.
            for f in self.categorical_features:         # f is an int (column index), categorical_features (and ordinal_features) must be list of int

                if f in self.ordinal_features:
                    # For ordinal features, create <= or > threshold tests for each value.
                    for v in self.categorical_names[f]:
                        idx = len(mapping)
                        if base_instance_disc[f] <= v: # and v != self.categorical_names[f][-1]: #v != len(self.categorical_names[f]) - 1:
                            mapping[idx] = (f, 'leq', v)
                            # print("value of feature", f, ":", data_row[f], "condition added:", mapping[idx])
                            # names[idx] = '%s <= %s' % (self.feature_names[f], v)
                        elif base_instance_disc[f] > v:
                            mapping[idx] = (f, 'geq', v)
                            # print("value of feature", f, ":", data_row[f], "condition added:", mapping[idx])
                            # names[idx] = '%s > %s' % (self.feature_names[f], v)
                        
                else:
                    # For nominal features, create an equality test matching the reference value.
                    idx = len(mapping)
                    mapping[idx] = (f, 'eq', base_instance_disc[f])
        # print("Initial mapping", mapping)
        # Remove duplicate predicates (same feature, operator, value)
        seen = set()
        unique_mapping = {}
        for idx, (f, op, v) in mapping.items():
            key = (f, op, v)
            if key not in seen:
                seen.add(key)
                unique_mapping[len(unique_mapping)] = (f, op, v)
        mapping = unique_mapping

        print("mapping", mapping)
        
        # Define actual sampler
        def sample_fn(present, num_samples, compute_labels=True):
            """Consumes a list of predicate indices (`present`) and generates perturbed samples that respect those predicates,
            and optionally evaluates the model on them."""

            # Separate the predicates into equality, <=, and > conditions
            conditions_eq = {}
            conditions_leq = {}
            conditions_geq = {}

            # Convert integers in `present` back to actual predicates using `mapping`, split into condition types dictionaries
            for x in present:
                f, op, v = mapping[x]
                if op == 'eq':
                    conditions_eq[f] = v
                if op == 'leq':
                    # keep the tightest (lowest) upper bound
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    conditions_leq[f] = min(conditions_leq[f], v)
                if op == 'geq':
                    # keep the tightest (largest) lower bound
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)
            #print("Separated conditions - eq:", conditions_eq, "geq:", conditions_geq, "leq:", conditions_leq)

            # Sample `num_samples` rows from the training data satisfying the requested predicates
            raw_data = self.sample_from_train(
                conditions_eq, {}, conditions_geq, conditions_leq, num_samples)
            
            # inutile: d_raw_data = raw_data
            # It encodes each sampled row as a binary vector (`data`) indicating whether each predicate in `mapping` holds for that row
            data = np.zeros((num_samples, len(mapping)), int)       # Creates vector of zeros of shape (num_samples, len(mapping))
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (raw_data[:, f] == v).astype(int)
                if op == 'leq':
                    data[:, i] = (raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (raw_data[:, f] > v).astype(int)
            # This is the feature space used by the beam search in Anchors

            # print("Boolean matrix", data)
            # print("Raw data samples:", raw_data)
            
            # If compute labels is True, it evaluates the wrapped model on the sampled rows
            # and records whether the prediction matches `true_label`.
            labels = []
            if compute_labels:
                if mode == "standard":
                    # Original anchors: positives if same prediction as `true_label`
                    preds = predict_fn(raw_data)
                    # print("Predictions on samples:\n", preds)
                    labels = (preds == true_label).astype(int)      # It is a boolean array of shape (num_samples,) !!!!!
                    # print("Labels:", labels)
                elif mode == "conformal":
                    # Conformal anchors: positives if `true_label` is NOT in the prediction set
                    probs = predict_fn(raw_data)
                    labels = (probs[:, query_label] < (1 - qhat)).astype(int)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

            return raw_data, data, labels
        
        return sample_fn, mapping

    def explain_instance(self, data_row, classifier, mode="standard", query_label=None, qhat=None, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100,
                          max_anchor_size=None, desired_label=None, beam_size=10, predicate_mode="original", mean_instances=None, **kwargs):
        """Run the Anchor beam search on ``data_row`` and package the result."""
        
        # Build the perturbation sampler and predicate mapping for this instance.
        sample_fn, mapping = self.get_sample_fn(data_row, classifier, mode=mode, query_label=query_label, qhat=qhat, desired_label=desired_label, predicate_mode=predicate_mode, mean_instances=mean_instances)
        
        # Run Anchor beam search passing the sampling closure and the statistical guarantees
        # The beam search proposes combinations of predicates, uses `sample_fn` to estimate their precision and coverage, 
        # and returns the best anchor satisfying the user-supplied thresholds.
        exp, valid_anchors = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, beam_size=beam_size, max_anchor_size=max_anchor_size,
            mapping=mapping, enable_conflict_check=True,
            **kwargs)
        
        # Convert predicate indices into readable feature/value strings.
        self.add_names_to_exp(data_row, exp, mapping)

        exp['instance'] = data_row
        if mode == "conformal":
            probs = classifier.predict_proba(self.encoder_fn(data_row.reshape(1, -1)))
            exp['prediction'] = int(np.argmax(probs[0]))
        else:
            preds = classifier.predict(self.encoder_fn(data_row.reshape(1, -1)))
            exp['prediction'] = int(preds[0])

        # Convert all valid anchors to readable format
        for va in valid_anchors:
            self.add_names_to_exp(data_row, va, mapping)
            va['instance'] = data_row
            if mode == "conformal":
                va['prediction'] = exp['prediction']  # same class as main exp
            else:
                va['prediction'] = exp['prediction']

        # Wrap up the payload as an AnchorExplanation ready for rendering.
        explanation = anchor_explanation.AnchorExplanation('tabular', exp, self.as_html)
        
        return explanation, valid_anchors

    def add_names_to_exp(self, data_row, hoeffding_exp, mapping):
        """Attach human-readable strings to the predicates stored in ``hoeffding_exp``."""
        # TODO: precision recall is all wrong, coverage functions wont work
        # anymore due to ranges

        # Integer predicate indices selected by the beam search (before translation).
        idxs = hoeffding_exp['feature']
        # print("Indices of predicates:", idxs, "corresponding to predicates:", [mapping[idx] for idx in idxs])

        # Prepare a list to hold the readable names, and rewrite the feature list using mapping
        # so it contains raw column indices instead of predicate indices.
        hoeffding_exp['names'] = []
        hoeffding_exp['feature'] = [mapping[idx][0] for idx in idxs]        # Replace predicate indices list with true feature column indices list
        # print("Features in anchor:", hoeffding_exp['feature'])
        # Loop through the chosen predicates and, for every `>` or `<=` condition,
        # update `ordinal_ranges[f]` so it tracks the tightest lower and upper bounds seen for each ordinal feature `f`.
        ordinal_ranges = {}
        for idx in idxs:
            f, op, v = mapping[idx]
            # print(f, op, v)
            if op in ('geq', 'leq'):
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float('-inf'), float('inf')]
            if op == 'geq':
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == 'leq':
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)

        # Avoid emitting duplicate interval descriptions when both bounds exist.
        handled = set()
        for idx in idxs:
            f, op, v = mapping[idx]
            # Name of feature f
            fname = self.feature_names[f] 

            if op == 'eq':
                if np.isnan(v):
                    condition = f"{fname} = NaN"
                elif f in self.categorical_names:
                    condition = f"{fname} = {v:.2f}"
                else:
                    condition = f"{fname} = {v:.2f}"

            elif op in ('geq', 'leq'):
                if f in handled:
                    continue
                geq, leq = ordinal_ranges.get(f, (float('-inf'), float('inf')))
                # If feature f has both finite bounds, print them together as a bounded interval
                if geq > float('-inf') and leq < float('inf'):
                    condition = f"{geq:.2f} < {fname} ≤ {leq:.2f}"
                # If only lower bound exists
                elif geq > float('-inf'):
                    condition = f"{fname} > {geq:.2f}"
                # If only upper bound exists
                elif leq < float('inf'):
                    condition = f"{fname} ≤ {leq:.2f}"
                else:
                    condition = f"{fname}: unconstrained"
                handled.add(f)

            # else:
            #     condition = f"{fname} {op} {v}"

            hoeffding_exp['names'].append(condition)

            # if op == 'eq':
            #     # build strings like `Feature = Value`
            #     fname = f"{self.feature_names[f]} = "
            #     if np.isnan(v):  # <-- added to skip NaNs safely
            #         fname += "NaN"
            #     elif f in self.categorical_names:
            #         #v = int(v)
            #         for el in self.categorical_names[f]:
            #             #print("el:", el, "v:", v)
            #             #print(str(el)==str(v))
            #             if str(el) == str(v):
            #                 #print("el", el, "v", v)
            #                 pos_v = self.categorical_names[f].index(el)
            #                 #print(pos_v)
            #         #print(f, v)
            #         # print(self.categorical_names)
            #         # print(self.categorical_names[f])
            #         fname += str(self.categorical_names[f][pos_v])
            #     else:
            #         fname += f"{v:.2f}"
            # else:
            #     # Only emit interval descriptions once per feature.
            #     if f in handled:
            #         continue

            #     geq, leq = ordinal_ranges[f]
            #     fname = ''
            #     geq_val = ''
            #     leq_val = ''
            #     if geq > float('-inf'):
            #         #if geq == len(self.categorical_names[f]) - 1:
            #         #    geq = geq - 1
            #         name = geq #+ 1 #self.categorical_names[f][geq + 1]
            #         name = str(name)
            #         if '<' in name:
            #             geq_val = name.split()[0]
            #         elif '>' in name:
            #             geq_val = name.split()[-1]
            #     if leq < float('inf'):
            #         # print('pippo: ', f, leq)
            #         name = leq #self.categorical_names[f][leq]
            #         name = str(name) 
            #         if leq == 0:
            #             leq_val = name.split()[-1]
            #         elif '<' in name:
            #             leq_val = name.split()[-1]
            #     if leq and geq:
            #         fname = '%s < %s <= %s' % (geq, self.feature_names[f], leq)
            #     elif leq:
            #         fname = '%s <= %s' % (self.feature_names[f], leq)
            #     elif geq:
            #         fname = '%s > %s' % (self.feature_names[f], geq)
            #     handled.add(f)

            # hoeffding_exp['names'].append(fname)
