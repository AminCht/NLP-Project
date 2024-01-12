import glob
import math
import numpy as np
from nltk.stem import PorterStemmer
import os

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
    for word, count in word_counts.items():
        print(f"The word '{word}' appears {count} times in the list.")


def stemming(tokens):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(token) for token in tokens]
   # write_tokens('TextProcessing/stemming.txt', stemmed_words)
    return stemmed_words


def text_processing():
    file_path = 'TextProcessing/61085-0.txt'
    text_content = read_file(file_path)
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
                stemming(tokens)
            elif operation == 0:
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def spell_correction():
    print("You selected Option 2")


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


def read_test_set(directory_path):
    text_files = glob.glob(os.path.join(directory_path, '*.txt'))
    global files_name_list
    files_name_list = []
    content = []
    for file_path in text_files:
        with open(file_path, 'r') as file:
            files_name_list.append(file.name)
            file_text = file.read()
            file_text = file_text.lower()
            content.append(file_text)

    return content


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
            word_comp_prob = (word_count / (sum(words_dict.values()) + (len(all_train_set)+1)))
        list.append(word_comp_prob)
    answer = np.log(list)
    answer = np.sum(answer)
    return answer


def calculate_class(test_set, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob,
                    soc_prob, talk_prob, train_set_words):

    for i in range(len(test_set)):
        comp_test_set = test_set[i].split()
        comp_test_set = stemming(comp_test_set)
        word_comp_prob = calculate_pc(comp_test_set, comp_dict, comp_prob, train_set_words)
        word_rec_prob = calculate_pc(comp_test_set, rec_dict, rec_prob, train_set_words)
        word_sci_prob = calculate_pc(comp_test_set, sci_dict, sci_prob, train_set_words)
        word_soc_prob = calculate_pc(comp_test_set, soc_dict, soc_prob, train_set_words)
        word_talk_prob = calculate_pc(comp_test_set, talk_dict, talk_prob, train_set_words)
        if word_comp_prob == max(word_comp_prob, word_rec_prob, word_sci_prob, word_soc_prob, word_talk_prob):
            print(f"{files_name_list[i]} belong to comp")
        elif word_rec_prob == max(word_comp_prob, word_rec_prob, word_sci_prob, word_soc_prob, word_talk_prob):
            print(f"{files_name_list[i]} belong to rec")
        elif word_sci_prob == max(word_comp_prob, word_rec_prob, word_sci_prob, word_soc_prob, word_talk_prob):
            print(f"{files_name_list[i]} belong to sci")
        elif word_soc_prob == max(word_comp_prob, word_rec_prob, word_sci_prob, word_soc_prob, word_talk_prob):
            print(f"{files_name_list[i]} belong to soc")
        else:
            print(f"{files_name_list[i]} belong to talk")


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
    test = read_test_set('Classification/Comp.graphics/test')
    calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob, soc_prob,
                    talk_prob, train_set_count)
    test = read_test_set('Classification/rec.autos/test')
    calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob, soc_prob,
                    talk_prob, train_set_count)
    test = read_test_set('Classification/sci.electronics/test')
    calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob, soc_prob,
                    talk_prob, train_set_count)
    test = read_test_set('Classification/soc.religion.christian/test')
    calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob, soc_prob,
                    talk_prob, train_set_count)
    test = read_test_set('Classification/talk.politics.mideast/test')
    calculate_class(test, comp_dict, rec_dict, sci_dict, soc_dict, talk_dict, comp_prob, rec_prob, sci_prob, soc_prob,
                    talk_prob, train_set_count)


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
