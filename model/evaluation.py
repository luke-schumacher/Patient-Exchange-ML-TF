import os
import pickle
import numpy as np
import tensorflow as tf
import Levenshtein
from pprint import pprint
from contextlib import redirect_stdout
import matplotlib.pyplot as plt


# Updated Function: Remove START and END Tokens
def remove_start_end_tokens(list_of_sequences):
    """
    Removes the START and END tokens from each sequence.
    Works with sequences of integers.
    """
    return [seq[1:-1] for seq in list_of_sequences]


# Updated Function: Compare Pair of Sequences
def score_pair_of_sequences(seq_a, seq_b):
    """
    Computes similarity score between two sequences of `sourceID` values.
    Uses Levenshtein distance on string representations of sequences.
    """
    string_a = " ".join(map(str, seq_a))
    string_b = " ".join(map(str, seq_b))
    return Levenshtein.ratio(string_a, string_b)


# Updated Function: Load Results
def load_results_from_txt(file_path):
    """
    Loads translation results from results.txt.
    """
    results = {}
    with open(file_path, "r") as f:
        exec(f"results = {f.read()}", {}, results)
    return results


# Update: Compare Data Sets
def compare_data_sets(input_data_dict,
                      input_data_dict_2,
                      score_function,
                      remove_pseudo_tokens=(False, True),
                      save_folder_path="",
                      names=("train", "model")):
    mean_scores_a = []
    mean_scores_b = []
    frequency_metrics = []

    sorted_conditions = list(sorted(list(input_data_dict.keys())))

    for cond in sorted_conditions:
        examples_a = input_data_dict[cond]
        examples_b = input_data_dict_2.get(cond, [])

        if remove_pseudo_tokens[0]:
            examples_a = remove_start_end_tokens(examples_a)

        if remove_pseudo_tokens[1]:
            examples_b = remove_start_end_tokens(examples_b)

        unique_examples_a = list(set(tuple(x) for x in examples_a))
        unique_examples_b = list(set(tuple(x) for x in examples_b))

        all_examples = list(set(unique_examples_a + unique_examples_b))
        sorted_examples = sorted(all_examples, key=len)

        frequencies_a = {x: examples_a.count(list(x)) / len(examples_a) for x in unique_examples_a}
        frequencies_b = {x: examples_b.count(list(x)) / len(examples_b) for x in unique_examples_b}

        for example in sorted_examples:
            if example not in frequencies_a:
                frequencies_a[example] = 0.0

            if example not in frequencies_b:
                frequencies_b[example] = 0.0

        frequencies_list_a = [frequencies_a[item] for item in sorted_examples]
        frequencies_list_b = [frequencies_b[item] for item in sorted_examples]

        frequency_mae = np.mean(np.abs(np.asarray(frequencies_list_a) - np.asarray(frequencies_list_b)))
        frequency_metrics.append(frequency_mae)

        best_scores_b = {
            item_b: max(score_function(item_b, item_a) for item_a in unique_examples_a)
            for item_b in unique_examples_b
        }
        best_scores_a = {
            item_a: max(score_function(item_a, item_b) for item_b in unique_examples_b)
            for item_a in unique_examples_a
        }

        mean_score_a = np.mean(list(best_scores_a.values()))
        mean_score_b = np.mean(list(best_scores_b.values()))

        mean_scores_a.append(mean_score_a)
        mean_scores_b.append(mean_score_b)

    return mean_scores_a, mean_scores_b, frequency_metrics


# Updated Main Function
def main(n_conditions=-1,
         n_samples=100,
         checkpoint_name="test_transformer",
         compute_shifted_examples=False):
    # Load evaluation data
    evaluation_data_path = os.path.join(checkpoint_name, "raw_evaluation_data.pkl")
    if not os.path.exists(evaluation_data_path):
        raise FileNotFoundError("Evaluation data not found. Generate it first.")

    with open(evaluation_data_path, "rb") as f:
        data_dict = pickle.load(f)

    # Load translation results from results.txt
    results_path = os.path.join(checkpoint_name, "results.txt")
    if not os.path.exists(results_path):
        raise FileNotFoundError("Translation results not found. Run the model and generate results.txt first.")

    print("Loading translation results...")
    translation_results = load_results_from_txt(results_path)

    # Ensure translation results are in the correct format
    for cond in translation_results:
        translation_results[cond] = [
            list(map(int, seq.split())) for seq in translation_results[cond] if seq
        ]

    # Compare data sets
    train_mean_scores, model_mean_scores, mean_freq_differences = compare_data_sets(
        data_dict,
        translation_results,
        score_function=score_pair_of_sequences,
        remove_pseudo_tokens=(False, True),
        save_folder_path=checkpoint_name,
        names=("train", "model")
    )

    # Output results
    print("Final scores:")
    print("Training vs Model:")
    print(f"Min, Mean, Max: {np.min(train_mean_scores)}, {np.mean(train_mean_scores)}, {np.max(train_mean_scores)}")
    print("Frequency differences:")
    print(f"Min, Mean, Max: {np.min(mean_freq_differences)}, {np.mean(mean_freq_differences)}, {np.max(mean_freq_differences)}")


if __name__ == '__main__':
    main()
