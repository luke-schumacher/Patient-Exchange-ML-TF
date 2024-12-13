import os
import numpy as np
import Levenshtein
from pprint import pprint
import matplotlib.pyplot as plt


# Helper: Load results from results.txt
def load_results_from_txt(file_path):
    """
    Loads translation results from results.txt.
    """
    results = {}
    with open(file_path, "r") as f:
        exec(f"results = {f.read()}", {}, results)
    return results


# Helper: Remove START and END Tokens
def remove_start_end_tokens(list_of_sequences):
    """
    Removes the START and END tokens from each sequence.
    Works with sequences of integers.
    """
    return [seq[1:-1] for seq in list_of_sequences]


# Scoring function for sequence similarity
def score_pair_of_sequences(seq_a, seq_b):
    """
    Computes similarity score between two sequences of `sourceID` values.
    Uses Levenshtein distance on string representations of sequences.
    """
    string_a = " ".join(map(str, seq_a))
    string_b = " ".join(map(str, seq_b))
    return Levenshtein.ratio(string_a, string_b)


# Frequency comparison and visualization
def make_frequency_plot(frequencies_a,
                        labels,
                        title,
                        save_path,
                        name="model"):
    """
    Plots frequencies of unique sequences in the results.
    """
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_axis = range(len(frequencies_a))

    ax.set_title(title)
    ax.set_xlabel("Unique Sequence")
    ax.set_ylabel("Frequency")
    ax.bar(x_axis, frequencies_a, color="blue", label=name)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Main Evaluation Function
def main(results_file="model/test_transformer/results.txt", checkpoint_name="model/test_transformer"):
    # Check for results.txt file
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"{results_file} not found. Please provide results.txt.")

    # Load results from results.txt
    print("Loading translation results...")
    translation_results = load_results_from_txt(results_file)

    # Flatten the results into one list
    all_sequences = []
    for key, sequences in translation_results.items():
        if isinstance(sequences, list):  # Ignore conditions like "A", "B" if they are empty lists
            all_sequences.extend(sequences)

    # Remove START and END tokens
    cleaned_sequences = remove_start_end_tokens(all_sequences)

    # Count frequencies of unique sequences
    unique_sequences = list(set(tuple(seq) for seq in cleaned_sequences))
    unique_sequences = sorted(unique_sequences, key=len)

    frequencies = {seq: cleaned_sequences.count(list(seq)) / len(cleaned_sequences) for seq in unique_sequences}
    frequency_list = [frequencies[seq] for seq in unique_sequences]

    # Save frequency plot
    make_frequency_plot(frequency_list,
                        labels=[" ".join(map(str, seq)) for seq in unique_sequences],
                        title="Frequency of Unique Sequences",
                        save_path=os.path.join(checkpoint_name, "frequency_plot.png"),
                        name="model")

    # Output evaluation metrics
    print("\nEvaluation Results:")
    print(f"Total sequences: {len(cleaned_sequences)}")
    print(f"Unique sequences: {len(unique_sequences)}")
    print(f"Frequency plot saved to: {os.path.join(checkpoint_name, 'frequency_plot.png')}")

    # Print unique sequences with their frequencies
    print("\nUnique Sequences and Frequencies:")
    pprint(frequencies)


if __name__ == '__main__':
    main()
