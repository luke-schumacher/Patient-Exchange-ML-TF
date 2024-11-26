import os
import sys

import context

from pprint import pprint
from contextlib import redirect_stdout 

import pickle
import numpy as np
import tensorflow as tf
import Levenshtein

import matplotlib.pyplot as plt

from use_cases.transformer_all_included import generate_data, Transformer, ConditionMapper, SeqMapper, Translator, \
    RAW_STRINGS, add_start_end_tokens, make_same_length #change to personal folder
from use_cases.transformer_all_included import make_training_data, convert_training_data, masked_accuracy, masked_loss

import string

CHARS = string.printable


def remove_start_end_tokens(list_of_strings):
    output_list = []

    for item in list_of_strings:
        output_list.append(item[1:-1])

    return output_list


def cycle_conditions_in_data_set(conditional_data, shift=1):
    import copy

    data_conditions = list(sorted(list(conditional_data.keys())))
    cycled_conditions = copy.copy(data_conditions)

    for _ in range(shift):
        cycled_conditions.append(cycled_conditions.pop(0))

    new_data = dict()

    for row in range(len(data_conditions)):
        new_data[cycled_conditions[row]] = copy.copy(conditional_data[data_conditions[row]])

    return new_data


def score_pair_of_sequences(seq_a, seq_b):
    # alternative: use difflib / SequenceMatcher

    unique_a = list(sorted(set(seq_a)))
    unique_b = list(sorted(set(seq_b)))

    all_items = list(sorted(unique_a + unique_b))
    num_items = len(all_items)

    assert num_items <= len(CHARS)

    conversion_dict = dict(zip(all_items, CHARS[:num_items]))

    string_a = "".join([conversion_dict[item] for item in seq_a])
    string_b = "".join([conversion_dict[item] for item in seq_b])

    return Levenshtein.ratio(string_a, string_b)


def score_pair(string_a, string_b):
    return Levenshtein.ratio(string_a, string_b)


def make_frequency_plot(frequencies_a,
                        frequencies_b,
                        labels,
                        title,
                        save_path,
                        names=("train", "model")):
    # plots 
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    axs.set_facecolor("darkgrey")

    x_axis = range(len(frequencies_a))

    axs.set_title(title)
    axs.set_xlabel("Unique Sequence")
    axs.set_ylabel("Frequency")

    axs.set_ylim((-0.1, 1.1))

    axs.set_xticks(x_axis, labels=labels, rotation=45)

    axs.scatter(x_axis, frequencies_a, label=f"{names[0]} data", color="yellow")
    axs.scatter(x_axis, frequencies_b, label=f"{names[1]} data", color="blue")
    axs.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def make_scores_plot(scores_a,
                     scores_b,
                     frequency_metrics,
                     labels,
                     save_path,
                     names=("train", "model")):

    scores_stats_a = np.min(scores_a), np.mean(scores_a), np.max(scores_a)
    scores_stats_b = np.min(scores_b), np.mean(scores_b), np.max(scores_b)
    frequency_stats = np.min(frequency_metrics), np.mean(frequency_metrics), np.max(frequency_metrics)

    fig, axs = plt.subplots(1, 1, figsize=(12, 5))

    axs.set_facecolor("darkgrey")

    axs.set_xlabel("Unique Condition")
    axs.set_ylabel("Score")

    x_axis = range(len(labels))
    axs.set_xticks(x_axis, labels=labels)

    axs.plot(x_axis, [1.0] * len(x_axis), color="black", linestyle="dotted")

    axs.plot(x_axis, 1 - np.asarray(frequency_metrics), color="red", label="freq scores")
    axs.plot(x_axis, 1 - np.asarray([frequency_stats[1]] * len(x_axis)), color="red", linestyle="dotted")

    axs.plot(x_axis, scores_a, color="yellow", label=f"{names[0]} scores")
    axs.plot(x_axis, [scores_stats_a[1]] * len(x_axis), color="yellow", linestyle="dotted")

    axs.plot(x_axis, scores_b, color="blue", label=f"{names[1]} scores")
    axs.plot(x_axis, [scores_stats_b[1]] * len(x_axis), color="blue", linestyle="dotted")

    axs.legend()
    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


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
        examples_b = input_data_dict_2[cond]

        if remove_pseudo_tokens[0]:
            examples_a = remove_start_end_tokens(examples_a)

        if remove_pseudo_tokens[1]:
            examples_b = remove_start_end_tokens(examples_b)

        unique_examples_a = list(set(examples_a))
        unique_examples_b = list(set(examples_b))

        all_examples = list(set(unique_examples_a + unique_examples_b))
        sorted_examples = sorted(all_examples, key=len)

        frequencies_a = {x: examples_a.count(x) / len(examples_a) for x in unique_examples_a}
        frequencies_b = {x: examples_b.count(x) / len(examples_b) for x in unique_examples_b}

        for example in sorted_examples:
            if example not in frequencies_a:
                frequencies_a[example] = 0.0

            if example not in frequencies_b:
                frequencies_b[example] = 0.0

        frequencies_list_a = [frequencies_a[item] for item in sorted_examples]
        frequencies_list_b = [frequencies_b[item] for item in sorted_examples]

        frequency_mae = np.mean(np.abs(np.asarray(frequencies_list_a) - np.asarray(frequencies_list_b)))
        frequency_metrics.append(frequency_mae)

        most_similar_in_a = {
            item_b: np.argmax([score_function(item_b, item_a) for item_a in unique_examples_a]) for
            item_b in unique_examples_b}

        most_similar_in_b = {
            item_a: np.argmax([score_function(item_a, item_b) for item_b in unique_examples_b]) for
            item_a in unique_examples_a}

        b_a_pairs = {item_b: unique_examples_a[most_similar_in_a[item_b]] for item_b in unique_examples_b}
        a_b_pairs = {item_a: unique_examples_b[most_similar_in_b[item_a]] for item_a in unique_examples_a}

        # calculate similarity scores
        best_scores_b = {item_b: score_function(item_b, b_a_pairs[item_b]) for item_b in b_a_pairs}
        best_scores_a = {item_a: score_function(item_a, a_b_pairs[item_a]) for item_a in a_b_pairs}

        mean_score_a = np.mean(list(best_scores_a.values()))
        mean_score_b = np.mean(list(best_scores_b.values()))

        mean_scores_a.append(mean_score_a)
        mean_scores_b.append(mean_score_b)

        make_frequency_plot(frequencies_list_a,
                            frequencies_list_b,
                            sorted_examples,
                            title=f"Frequencies for Condition: {cond}",
                            save_path=os.path.join(save_folder_path, f"frequencies_{cond}.png"),
                            names=names)

    make_scores_plot(mean_scores_a,
                     mean_scores_b,
                     frequency_metrics,
                     sorted_conditions,
                     save_path=os.path.join(save_folder_path, f"scores.png"), names=names)

    return mean_scores_a, mean_scores_b, frequency_metrics


# --------------------------------------- main use case
def main(n_conditions=-1,
         n_samples=3,
         checkpoint_name="test_transformer",
         compute_shifted_examples=False):

    # get data
    evaluation_data_path = os.path.join(checkpoint_name, "raw_evaluation_data.pkl")

    if not os.path.exists(evaluation_data_path):
        print("Generating training like data.")
        
        conditions, sequences = generate_data(data_size=5000)

        data_dict = {}
        for i, cond in enumerate(conditions):
            if cond not in data_dict:
                data_dict[cond] = []

            data_dict[cond].append(sequences[i])

        with open(evaluation_data_path, "wb") as f:
            pickle.dump(data_dict, f)
    else:
        print("Loading saved training like data.")
        
        with open(evaluation_data_path, "rb") as f:
            data_dict = pickle.load(f)

    # translate examples
    results_path = os.path.join(checkpoint_name, "evaluation_results.pkl")

    if not os.path.exists(results_path):
        print("Generating data with model.")
        
        # build transformer
        if not os.path.exists(checkpoint_name):
            raise ValueError()
        else:
            if os.path.exists(os.path.join(checkpoint_name, "encoder.weights.h5")):
                cond_mapper = ConditionMapper()
                seq_mapper = SeqMapper()

                conditions, sequences = generate_data(data_size=5)
                conditions, sequences = add_start_end_tokens(conditions, sequences)
                conditions, sequences = make_same_length(conditions, sequences)
                raw_model_input_data, raw_model_output_data = make_training_data(conditions, sequences)
                model_input_data, model_output_data = convert_training_data(raw_model_input_data,
                                                                            raw_model_output_data,
                                                                            mappers=(cond_mapper, seq_mapper))

                transformer = Transformer(
                    num_layers=3,
                    d_model=32,
                    num_heads=8,
                    dff=128,
                    input_vocab_size=cond_mapper.dimension,
                    target_vocab_size=seq_mapper.dimension,
                    dropout_rate=0.1,
                )

                transformer(model_input_data)
                transformer.summary()

                transformer.load_model_from_weights(checkpoint_name)

                # compile transformer
                transformer.compile(
                    loss=masked_loss,
                    optimizer='Adam',
                    metrics=[masked_accuracy])

                # build translator
                translator = Translator(transformer, mappers=(cond_mapper, seq_mapper))

                print("Model loaded successfully!")
                print("\n")
            else:
                raise ValueError()

        # generate results
        translation_results = dict()

        if n_conditions == -1:
            strings_to_use = RAW_STRINGS
        elif 0 < n_conditions < len(RAW_STRINGS):
            strings_to_use = RAW_STRINGS[:n_conditions]
        else:
            raise NotImplementedError()

        for char in strings_to_use:

            print(f"Generating output for condition {char}")

            translation_results[char] = []
            example_cond = cond_mapper.map_to_ints(char)

            for i in range(n_samples):
                model_text, _ = translator(example_cond)

                print(model_text)

                translation_results[char].append(model_text)

        with open(results_path, "wb") as f:
            pickle.dump(translation_results, f)
    else:
        
        print("Loading saved model generated data.")
        
        with open(results_path, "rb") as f:
            translation_results = pickle.load(f)

    # evaluation

    if compute_shifted_examples:
        for i in range(len(data_dict)):
            print(f"Shifting {i} steps")

            temp_checkpoint = os.path.join(checkpoint_name, f"cycle_{i}")

            if not os.path.exists(temp_checkpoint):
                os.mkdir(temp_checkpoint)

            cycled_data_dict = cycle_conditions_in_data_set(data_dict, shift=i)
            
            _ = compare_data_sets(data_dict,
                                cycled_data_dict,
                                score_pair,
                                remove_pseudo_tokens=(False, False),
                                save_folder_path=temp_checkpoint,
                                names=("original", f"shifted {i}"))

    train_mean_scores, model_mean_scores, mean_freq_differences = compare_data_sets(data_dict,
                                                                                    translation_results,
                                                                                    score_function=score_pair,
                                                                                    remove_pseudo_tokens=(False, True),
                                                                                    save_folder_path=checkpoint_name,
                                                                                    names=("train", "model"))

    print("Final scores:")
    print("min, mean, max of mean best scores")

    train_scores = np.min(train_mean_scores), np.mean(train_mean_scores), np.max(train_mean_scores)
    model_scores = np.min(model_mean_scores), np.mean(model_mean_scores), np.max(model_mean_scores)
    freq_scores = np.min(mean_freq_differences), np.mean(mean_freq_differences), np.max(mean_freq_differences)

    print("Comparison training examples vs. model examples:")
    print(f"scores: {train_scores}")
    print(f"i.e.:")
    print(f"min == 1.0 means: every train example has a corresponding model example")
    print("i.e.: the model generates ALL training examples")

    print("Comparison model examples vs. training examples:")
    print(f"scores: {model_scores}")
    print(f"i.e.:")
    print(f"min < 1.0 means: there are model examples without exact correspondence in training data")
    print(f"max == 1.0 means: there are model examples with exact correspondence in training data")
    print("i.e.: the model generates data that was not in the training data")

    print(f"mean ~= 1: the model generates same outputs, but slightly different")

    print("Comparison of frequency differences")
    print(f"frequency differences: {freq_scores}")


if __name__ == '__main__':
    main(n_conditions=-1,
         n_samples=100,
         checkpoint_name="test_transformer",
         compute_shifted_examples=False,
         )
