import os
import re

def remove_comments_and_special_chars(java_code):
    # Remove single-line comments
    java_code = re.sub(r'//.*', '', java_code)
    # Remove multi-line comments
    java_code = re.sub(r'/\*.*?\*/', '', java_code, flags=re.DOTALL)
    return java_code

def extract_methods(java_code):
    methods = []
    method_pattern = re.compile(r'(public|protected|private|static|\s)*[\w<>,\[\]]+\s+\w+\s*\([^\)]*\)\s*\{', re.DOTALL)
    
    start_positions = []
    braces_stack = 0
    i = 0
    while i < len(java_code):
        match = method_pattern.search(java_code, i)
        if match:
            start_positions.append(match.start())
            braces_stack = 1  
            i = match.end()
            while braces_stack > 0 and i < len(java_code):
                if java_code[i] == '{':
                    braces_stack += 1
                elif java_code[i] == '}':
                    braces_stack -= 1
                i += 1
            if braces_stack == 0:
                methods.append(java_code[start_positions.pop():i])
        else:
            break
    return methods

def process_java_files(directory, output_file):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    with open(output_file, 'w') as master_file:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.java'):
                    java_file_path = os.path.join(root, file)
                    with open(java_file_path, 'r') as java_file:
                        java_code = java_file.read()
                        cleaned_code = remove_comments_and_special_chars(java_code)
                        methods = extract_methods(cleaned_code)
                        for method in methods:
                            master_file.write(method.strip() + '\n\n')

if __name__ == "__main__":
    directory = "input/collected_dataset/java_source"  
    output_file = "extracted_java.txt"  

    process_java_files(directory, output_file)
    print(f"Methods have been extracted and saved to {output_file}")
