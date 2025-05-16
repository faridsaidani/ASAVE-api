import json

def jsonl_to_json(jsonl_file_path, json_file_path):
    """
    Converts a JSONL file to a JSON file containing an array of objects.
    
    Args:
        jsonl_file_path (str): Path to the input JSONL file
        json_file_path (str): Path to save the output JSON file
    """
    # Read JSONL file and parse each line
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        # Filter out comment lines and empty lines, parse each valid line as JSON
        json_objects = []
        for line in jsonl_file:
            line = line.strip()
            if line and not line.startswith('//'):
                try:
                    json_obj = json.loads(line)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:50]}... - {e}")
    
    # Write to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_objects, json_file, ensure_ascii=False, indent=2)
    
    return len(json_objects)

# Example usage
if __name__ == "__main__":
    input_path = "e:/Farid/ISDBI Onsite/asave_project/srma_frontend_output/mined_rules_SSRM1_SS10.jsonl"
    output_path = "e:/Farid/ISDBI Onsite/asave_project/srma_frontend_output/mined_rules_SSRM1_SS10.json"
    
    count = jsonl_to_json(input_path, output_path)
    print(f"Successfully converted {count} JSON objects from JSONL to JSON format")