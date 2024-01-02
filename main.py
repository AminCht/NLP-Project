from nltk.stem import PorterStemmer


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


def lowerCase_folding(content):
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
    write_tokens('TextProcessing/stemming.txt', stemmed_words)


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
                lowerCase_folding(text_content)
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


def text_classification():
    print("You selected Option 3")


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
