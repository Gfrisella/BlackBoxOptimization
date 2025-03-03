import numpy as np
import gzip
import pickle
import glob

def f(x):
    # Example function to process data
    return np.sum(x)

def split_data(X, n_chunks):
    """Splits X into n_chunks for parallel processing."""
    return np.array_split(X, n_chunks)

# Main script
if __name__ == "__main__":
    import sys
    # Read arguments passed from SLURM (chunk index and total chunks)
    chunk_index = int(sys.argv[1])
    n_chunks = int(sys.argv[2])

    input_file = 'Project/MuonsAndMatter/data/inputs.pkl'
    
    # Example data (replace this with your actual dataset)
    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)
    # Split data into chunks
    chunks = split_data(data, n_chunks)

    # Select the chunk for this SLURM task
    data_chunk = chunks[chunk_index]

    # Process the chunk
    result = f(data_chunk)

    # Save or print the result (e.g., write to a file)
    print(f"Chunk {chunk_index}: Result = {result}")



    # Collect and combine results
    results = []
    for file in sorted(glob.glob("output_*.out")):
        with open(file, "r") as f:
            results.append(float(f.readline().split("=")[-1].strip()))

    # Aggregate results
    final_result = np.sum(results)
    print("Final Result:", final_result)
