import glob
import ast
import math

import numpy as np
from nltk.stem import PorterStemmer
import os
import enchant

comp_train_set_count = 0
rec_train_set_count = 0
sci_train_set_count = 0
soc_train_set_count = 0
talk_train_set_count = 0
files_name_list = []


def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            content = content.strip()
            return content
    except FileNotFoundError:
        print(f"The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def write_file(output_file_path, content):
    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(content)
    except Exception as e:
        print(f"An error occurred: {e}")


def write_tokens(output_file_path, tokens):
    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for item in tokens:
                output_file.write(item + '\n')
    except Exception as e:
        print(f"An error occurred: {e}")


def print_menu():
    print("1. Text Processing")
    print("2. Spell Correction")
    print("3. Text Classification")


def print_text_processing_menu():
    print("1. Tokenization")
    print("2. LowerCase")
    print("3. Calculate Tokens Count")
    print("4. Stemming")


def print_classification_menu():
    print("1. Make Dic")
    print("2. Calculate Probability(p(C))")
    print("3. P(w|c)")
    print("4. Stemming")


def tokenization(content):
    punctuation = set('!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    for char in punctuation:
        content = content.replace(char, ' ')
    tokens = content.split()
    for i in range(len(tokens)):
        if tokens[i] == 'll':
            tokens[i] = 'will'
        elif tokens[i] == 're':
            tokens[i] = 'are'
    i = 0
    while i < len(tokens):
        if tokens[i] == 's' and tokens[i - 1] not in ['he', 'she', 'it']:
            del tokens[i]
        else:
            i += 1
    return tokens


def lowercase_folding(content):
    output_file_path = 'TextProcessing/Lowercase.txt'
    content = content.lower()
    write_file(output_file_path, content)


def tokens_count(tokens):
    word_counts = {}
    for token in tokens:
        if token in word_counts:
            word_counts[token] += 1
        else:
            word_counts[token] = 1
    # for word, count in word_counts.items():
    #     print(f"The word '{word}' appears {count} times in the list.")
    write_file('TextProcessing/Tokens_count.txt', str(word_counts))


def stemming(tokens):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(token) for token in tokens]
    # write_tokens('TextProcessing/stemming.txt', stemmed_words)
    return stemmed_words


def text_processing():
    file_path = 'TextProcessing/61085-0.txt'
    text_content = read_file(file_path)
    text_content = text_content.lstrip('\ufeff')
    while True:
        print_text_processing_menu()
        tokens = tokenization(text_content)
        try:
            operation = int(input("Enter your choice (1-3, or 0 to exit): "))
            if operation == 1:
                write_tokens('TextProcessing/tokens.txt', tokens)
            elif operation == 2:
                lowercase_folding(text_content)
            elif operation == 3:
                tokens_count(tokens)
            elif operation == 4:
                stemmed_words = stemming(tokens)
                write_file('TextProcessing/stemmed.txt', str(stemmed_words))
            elif operation == 0:
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def damerau_levenshtein_distance(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)

    # Create a matrix to store the distances
    d = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

    # Initialize the matrix
    for i in range(len_str1 + 1):
        d[i][0] = i
    for j in range(len_str2 + 1):
        d[0][j] = j

    # Fill the matrix
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # Deletion
                d[i][j - 1] + 1,  # Insertion
                d[i - 1][j - 1] + cost,  # Substitution
            )
            if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)  # Transposition

    return d[len_str1][len_str2]


def candidate_word(misspell_word):
    english_dict = enchant.Dict("en_US")
    suggestions = english_dict.suggest(misspell_word)

    return suggestions


def deletion(miss_spell, word):
    for index in range(max(len(miss_spell), len(word))):
        if index != len(miss_spell) - 1:
            if miss_spell[index] != word[index] and index != 0:
                y = word[index]
                x = word[index - 1]
                return x, y, 0
            elif miss_spell[index] != word[index] and index == 0:
                y = word[index]
                x = word[index + 1]
                return x, y, 0
        if index == len(miss_spell) - 1:
            y = word[index + 1]
            x = miss_spell[index]
            return x, y, 0


def insertion(miss_spell, word):
    for index in range(max(len(miss_spell), len(word))):
        if index != len(word) - 1:
            if miss_spell[index] != word[index] and index != 0:
                y = miss_spell[index]
                x = miss_spell[index - 1]
                return x, y, 1
            elif miss_spell[index] != word[index] and index == 0:
                y = miss_spell[index]
                x = miss_spell[index + 1]
                return x, y, 1
        if index == len(word) - 1:
            y = miss_spell[index]
            x = miss_spell[index - 1]
            return x, y, 1


def sub_trans(miss_spell, word):
    for index in range(len(miss_spell)):
        if index != len(word) - 1:
            if miss_spell[index] != word[index]:
                if miss_spell[index + 1] != word[index + 1]:
                    y = miss_spell[index + 1]
                    x = miss_spell[index]
                    return x, y, 3
                else:
                    y = word[index]
                    x = miss_spell[index]
                    return x, y, 2
        if index == len(word) - 1:
            if miss_spell[index] != word[index]:
                y = word[index]
                x = miss_spell[index]
                return x, y, 2


def check_type_of_confusion(miss_spell, word):
    y = len(miss_spell) - len(word)
    if y == -1:
        x, y, z = deletion(miss_spell, word)
    elif y == 1:
        x, y, z = insertion(miss_spell, word)
    elif y == 0:
        x, y, z = sub_trans(miss_spell, word)
    return x, y, z


def noisy_channel_prob(x, y, z, dataset, del_confusion_matrix, ins_confusion_matrix, sub_confusion_matrix,
                       trans_confusion_matrix):
    if z == 0:
        conf_matrix_col = x + y
        prob = del_confusion_matrix[conf_matrix_col] / sum(conf_matrix_col in element for element in dataset)
    elif z == 1:
        conf_matrix_col = x + y
        prob = ins_confusion_matrix[conf_matrix_col] / sum(x in element for element in dataset)
    elif z == 2:
        conf_matrix_col = x + y
        prob = sub_confusion_matrix[conf_matrix_col] / sum(y in element for element in dataset)
    elif z == 3:
        conf_matrix_col = x + y
        prob = trans_confusion_matrix[conf_matrix_col] / sum(conf_matrix_col in element for element in dataset)
    return prob


def spell_correction():
    words = read_file('./Spell Correction/spell-errors.txt')
    real_words = read_file('./Spell Correction/test/Dictionary/dictionary.data')
    dataset = read_file('./Spell Correction/test/Dictionary/Dataset.data')
    dataset = dataset.split()
    misspell_words = read_file('./Spell Correction/test/spell-testset.txt')
    misspell_words = misspell_words.split()
    del_confusion_matrix = read_file('./Spell Correction/test/Confusion Matrix/del-confusion.data')
    ins_confusion_matrix = read_file('./Spell Correction/test/Confusion Matrix/ins-confusion.data')
    sub_confusion_matrix = read_file('./Spell Correction/test/Confusion Matrix/sub-confusion.data')
    trans_confusion_matrix = read_file('./Spell Correction/test/Confusion Matrix/Transposition-confusion.data')
    del_confusion_matrix = ast.literal_eval(del_confusion_matrix)
    ins_confusion_matrix = ast.literal_eval(ins_confusion_matrix)
    sub_confusion_matrix = ast.literal_eval(sub_confusion_matrix)
    trans_confusion_matrix = ast.literal_eval(trans_confusion_matrix)
    real_words = real_words.split()
    dataset_words_count = words_count(dataset)
    correct_wordss = ''
    for mis_spell in misspell_words:
        candidate_words = candidate_word(mis_spell)
        main_candidate = []
        if mis_spell in real_words:
            main_candidate.append(mis_spell)
        for candid in candidate_words:
            candid = candid.lower()
            distance = damerau_levenshtein_distance(mis_spell, candid)
            if distance == 1 and ' ' not in candid and '-' not in candid:
                main_candidate.append(candid)
        if len(main_candidate) == 0:
            correct_wordss += f'corrected of {mis_spell} is {mis_spell}' + f'\n'
            continue
        correct_word = {}
        for candid in main_candidate:
            x, y, z = check_type_of_confusion(mis_spell, candid)
            prob = noisy_channel_prob(x, y, z, dataset, del_confusion_matrix, ins_confusion_matrix,
                                      sub_confusion_matrix, trans_confusion_matrix)
            if candid in dataset_words_count:
                prob *= (dataset_words_count[candid]) / sum(dataset_words_count.values())
            else:
                prob *= (1 / sum(dataset_words_count.values()))
            prob *= 10 ** 9
            correct_word[candid] = prob
        max_key = max(correct_word, key=correct_word.get)
        print(f'corrected of {mis_spell} is ' + max_key)
        correct_wordss += f'corrected of {mis_spell} is {max_key}' + f'\n'
    write_file('./Spell Correction/output.txt', correct_wordss)


def read_classification_files(directory_path):
    text_files = glob.glob(os.path.join(directory_path, '*.txt'))
    combined_content = ''
    if "Comp" in directory_path:
        global comp_train_set_count
        comp_train_set_count = len(text_files)
    elif "rec" in directory_path:
        global rec_train_set_count
        rec_train_set_count = len(text_files)
    elif "sci" in directory_path:
        global sci_train_set_count
        sci_train_set_count = len(text_files)
    elif "soc" in directory_path:
        global soc_train_set_count
        soc_train_set_count = len(text_files)
    else:
        global talk_train_set_count
        talk_train_set_count = len(text_files)
    for file_path in text_files:
        with open(file_path, 'r') as file:
            content = file.read()
            content = content.lower()
            combined_content += content + '\n'

    return combined_content


def read_test_set(directory_path, test_sets_count):
    text_files = glob.glob(os.path.join(directory_path, '*.txt'))
    test_sets_count += len(text_files)
    global files_name_list
    files_name_list = []
    content = []
    for file_path in text_files:
        with open(file_path, 'r') as file:
            files_name_list.append(file.name)
            file_text = file.read()
            file_text = file_text.lower()
            content.append(file_text)

    return content, test_sets_count


def words_count(tokens):
    word_counts = {}
    for token in tokens:
        if token in word_counts:
            word_counts[token] += 1
        else:
            word_counts[token] = 1
    # for word, count in word_counts.items():
    #     print(f"The word '{word}' appears {count} times in the list.")
    return word_counts


def calculate_class_probabilities():
    comp_probability = comp_train_set_count / (
            comp_train_set_count + rec_train_set_count + soc_train_set_count + sci_train_set_count + talk_train_set_count)
    rec_probability = rec_train_set_count / (
            comp_train_set_count + rec_train_set_count + soc_train_set_count + sci_train_set_count + talk_train_set_count)
    sci_probability = sci_train_set_count / (
            comp_train_set_count + rec_train_set_count + soc_train_set_count + sci_train_set_count + talk_train_set_count)
    soc_probability = soc_train_set_count / (
            comp_train_set_count + rec_train_set_count + soc_train_set_count + sci_train_set_count + talk_train_set_count)
    talk_probability = talk_train_set_count / (
            comp_train_set_count + rec_train_set_count + soc_train_set_count + sci_train_set_count + talk_train_set_count)

    # print(f"P(comp) is {round(comp_probability, 2)}")
    # print(f"P(rec) is {round(rec_probability, 2)}")
    # print(f"P(sci) is {round(sci_probability, 2)}")
    # print(f"P(soc) is {round(soc_probability, 2)}")
    # print(f"P(talk) is {round(talk_probability, 2)}")
    return round(comp_probability, 2), round(rec_probability, 2), round(sci_probability, 2), round(soc_probability, 2), \
           round(talk_probability, 2)


def calculate_pc(test_set, words_dict, prob, all_train_set):
    list = []
    word_comp_prob = prob
    list.append(word_comp_prob)
    for word in test_set:
        if word in words_dict:
            word_count = words_dict[word]
        else:
            word_count = 1
        if word in all_train_set:
            word_comp_prob = (word_count / (sum(words_dict.values()) + len(all_train_set)))
        else:
            word_comp_prob = (word_count / (sum(words_dict.values()) + (len(all_train_set) + 1)))
        list.append(word_comp_prob)
    answer = np.log(list)
    answer = np.sum(answer)
    return answer


def calculate_class(test_sets, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob,
                    soc_prob, talk_prob, train_set_words, corrected_class, tp):
    for i in range(len(test_sets)):
        test_set = test_sets[i].split()
        test_set = stemming(test_set)
        word_comp_prob = calculate_pc(test_set, comp_dict, comp_prob, train_set_words)
        word_rec_prob = calculate_pc(test_set, rec_dict, rec_prob, train_set_words)
        word_sci_prob = calculate_pc(test_set, sci_dict, sci_prob, train_set_words)
        word_soc_prob = calculate_pc(test_set, soc_dict, soc_prob, train_set_words)
        word_talk_prob = calculate_pc(test_set, talk_dict, talk_prob, train_set_words)
        if word_comp_prob == max(word_comp_prob, word_rec_prob, word_sci_prob, word_soc_prob, word_talk_prob):
            print(f"{files_name_list[i]} belong to comp")
            if corrected_class == 'Comp':
                tp += 1
        elif word_rec_prob == max(word_comp_prob, word_rec_prob, word_sci_prob, word_soc_prob, word_talk_prob):
            print(f"{files_name_list[i]} belong to rec")
            if corrected_class == 'rec':
                tp += 1
        elif word_sci_prob == max(word_comp_prob, word_rec_prob, word_sci_prob, word_soc_prob, word_talk_prob):
            print(f"{files_name_list[i]} belong to sci")
            if corrected_class == 'sci':
                tp += 1
        elif word_soc_prob == max(word_comp_prob, word_rec_prob, word_sci_prob, word_soc_prob, word_talk_prob):
            print(f"{files_name_list[i]} belong to soc")
            if corrected_class == 'soc':
                tp += 1
        else:
            print(f"{files_name_list[i]} belong to talk")
            if corrected_class == 'talk':
                tp += 1
    return tp


def text_classification():
    comp_train_set = read_classification_files('Classification/Comp.graphics/train')
    rec_train_set = read_classification_files('Classification/rec.autos/train')
    sci_train_set = read_classification_files('Classification/sci.electronics/train')
    soc_train_set = read_classification_files('Classification/soc.religion.christian/train')
    talk_train_set = read_classification_files('Classification/talk.politics.mideast/train')
    comp_train_set = comp_train_set.split()
    rec_train_set = rec_train_set.split()
    sci_train_set = sci_train_set.split()
    soc_train_set = soc_train_set.split()
    talk_train_set = talk_train_set.split()
    all_train_sets = comp_train_set + rec_train_set + sci_train_set + soc_train_set + talk_train_set
    comp_train_set = stemming(comp_train_set)
    rec_train_set = stemming(rec_train_set)
    sci_train_set = stemming(sci_train_set)
    soc_train_set = stemming(soc_train_set)
    talk_train_set = stemming(talk_train_set)
    comp_dict = words_count(comp_train_set)
    rec_dict = words_count(rec_train_set)
    sci_dict = words_count(sci_train_set)
    soc_dict = words_count(soc_train_set)
    talk_dict = words_count(talk_train_set)
    train_set_count = words_count(all_train_sets)
    comp_prob, rec_prob, sci_prob, soc_prob, talk_prob = calculate_class_probabilities()
    tp = 0
    test_sets_count = 0
    test, test_sets_count = read_test_set('Classification/Comp.graphics/test', test_sets_count)
    tp = calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob,
                         soc_prob,
                         talk_prob, train_set_count, 'Comp', tp)
    test, test_sets_count = read_test_set('Classification/rec.autos/test', test_sets_count)
    tp = calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob,
                         soc_prob,
                         talk_prob, train_set_count, 'rec', tp)
    test, test_sets_count = read_test_set('Classification/sci.electronics/test', test_sets_count)
    tp = calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob,
                         soc_prob,
                         talk_prob, train_set_count, 'sci', tp)
    test, test_sets_count = read_test_set('Classification/soc.religion.christian/test', test_sets_count)
    tp = calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob,
                         soc_prob,
                         talk_prob, train_set_count, 'soc', tp)
    test, test_sets_count = read_test_set('Classification/talk.politics.mideast/test', test_sets_count)
    tp = calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob,
                         soc_prob,
                         talk_prob, train_set_count, 'talk', tp)
    accuracy = tp / test_sets_count

    print(f'accuracy is {accuracy * 100}')


if __name__ == '__main__':
    while True:
        print_menu()
        try:
            choice = int(input("Enter your choice (1-3, or 0 to exit): "))

            if choice == 1:
                text_processing()
            elif choice == 2:
                spell_correction()
            elif choice == 3:
                text_classification()
            elif choice == 0:
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
