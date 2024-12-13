import re
import random

def inject_bugs(code):
    def mutate_comparisons(line):
        return re.sub(r'(==|!=|>|<|>=|<=)', lambda m: random.choice(['==', '!=', '>', '<', '>=', '<=']), line)

    def mutate_constants(line):
        return re.sub(r'\b\d+\b', lambda m: str(int(m.group()) + random.randint(-5, 5)), line)

    def remove_or_add_semicolon(line):
        if ';' in line and random.random() < 0.5:
            return line.replace(';', '', 1)
        elif random.random() < 0.3:
            return line + ';'
        return line

    def break_variable_names(line):
        return re.sub(r'\b(\w+)\b', lambda m: m.group(1) + '_bug' if random.random() < 0.1 else m.group(1), line)

    def flip_boolean_literals(line):
        return re.sub(r'\b(true|false)\b', lambda m: 'false' if m.group() == 'true' else 'true', line)

    def delete_or_add_braces(line):
        if '{' in line and random.random() < 0.3:
            return line.replace('{', '', 1)
        elif random.random() < 0.2:
            return line + ' {'
        return line

    mutations = [
        mutate_comparisons,
        mutate_constants,
        remove_or_add_semicolon,
        break_variable_names,
        flip_boolean_literals,
        delete_or_add_braces,
    ]

    for mutation in random.sample(mutations, k=random.randint(1, 2)):
        code = mutation(code)
    return code

def ensure_function_has_bug(function_code):
    lines = function_code.split('\n')
    if len(lines) > 2:
        target_line = random.randint(1, len(lines) - 2)
        lines[target_line] = inject_bugs(lines[target_line])
    return '\n'.join(lines)

def process_java_file_with_bugs(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        function_buffer = []
        inside_function = False

        for line in infile:
            if '{' in line and not inside_function:
                inside_function = True
                function_buffer.append(line)
            elif '}' in line and inside_function:
                function_buffer.append(line)
                inside_function = False
                buggy_function = ensure_function_has_bug(''.join(function_buffer))
                outfile.write(buggy_function + '\n')
                function_buffer = []
            elif inside_function:
                function_buffer.append(line)
            else:

if __name__ == "__main__":
    input_file = "extracted_java.txt" 
    output_file = "mutated_java.txt"  
    process_java_file_with_bugs(input_file, output_file)
    print(f"Mutated Java code with bugs saved to {output_file}")
