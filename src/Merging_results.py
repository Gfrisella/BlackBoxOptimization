import pickle
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rm', action = 'store_true')
args = parser.parse_args()

# Path to the directory where the pickle files are stored
directory = '/home/hep/gfrise/Project/BlackBoxOptimization/Outputs'

# Initialize an empty list to store the dictionaries
dict_list = []

# Loop through all .pkl files in the directory
for filename in os.listdir(directory):
    if filename.startswith("results_rank_") and filename.endswith(".pkl"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as f:
            # Load the dictionary from the pickle file
            data = pickle.load(f)
            
            # Append the dictionary to the list
            dict_list.append(data)
        if args.rm:
            # Delete the pickle file after loading its content
            os.remove(file_path)


# Initialize an empty dictionary for the merged data
merged_data = {'name': None, 'rank': [], 'weight': None, 'loss_muons': 0.0, 'loss' : [], 'particle_type' : [], 'x': [], 'y': [], 'z': [], 'px': [], 'py': [], 'pz': []}

# Merge dictionaries
for data in dict_list:
    # Keep the name from the first dictionary (assuming all have the same name)
    if data['name'] is not None:
        merged_data['name'] = data['name']
        
    if data['weight'] is not None:
        merged_data['weight'] = data['weight']
    
    if data['w'][0] is  not None:
        merged_data['w'] = data['w']    
    
    # Append the rank value
    merged_data['rank'].append(data['rank'])
    
    # Sum the muon_loss values
    merged_data['loss_muons'] += int(data['loss_muons'])

    # Sum the muon_loss values
    merged_data['loss'].append(data['loss'])
    
    merged_data['particle_type'] += data['particle_type']
    
    merged_data['x'] += data['x']
    merged_data['y'] += data['y']
    merged_data['z'] += data['z']
    merged_data['px'] += data['px']
    merged_data['py'] += data['py']
    merged_data['pz'] += data['pz']
    


try:
    output_directory = '/disk/users/gfrise/Project/BlackBoxOptimization/Outputs/'
    output_file = "merged_data.pkl"
    with open(output_directory + output_file, 'wb') as f:
        pickle.dump(merged_data, f)
except Exception as e:
    print(f"Failed to save combined results: {e}")

print(f"Saved merged data to: {output_directory+ output_file}")