import re

def tokenize_java_methods(input_file, output_file):
    def tokenize_code(code):
        tokenized = re.sub(r'([()\[\]{};.,<>!=+-/*&|~?:])', r' \1 ', code)
        tokenized = re.sub(r'\s+', ' ', tokenized).strip()
        return tokenized

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            tokenized_line = tokenize_code(line)
            outfile.write(tokenized_line + '\n')

if __name__ == "__main__":
    input_file = "extracted_java.txt"
    output_file = "tokenized_java.txt"
    tokenize_java_methods(input_file, output_file)
    print(f"Tokenized methods saved to {output_file}")
