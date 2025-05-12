import json

# Path to your JSON file
input_file = "/Users/bhavish/rbc_plt_annotations/train2017.json"
output_file = "/Users/bhavish/rbc_plt_annotations/updated_train_2017.json"

# Load the JSON data
with open(input_file, 'r') as file:
    data = json.load(file)

# Increment the ID for each annotation
new_id = 1
for annotation in data.get('annotations'):
    annotation['id'] = new_id
    new_id += 1

# Save the updated JSON data
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Updated annotations saved to {output_file}")