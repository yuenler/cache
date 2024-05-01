import pandas as pd

def address_to_cacheline(address, line_size=64):
    return (address // line_size)

def parse_file(file_path, line_size=64):
    with open(file_path, 'r') as file:
        cachelines = [
            address_to_cacheline(int(line.strip()), line_size)
            for line in file if line.strip().isdigit()
        ]
    return cachelines

def main():
    cache_line_size = 64

    file_paths = ['libquantum.out', 'hmmer.out']
    output_files = ['libquantum_cachelines.csv', 'hmmer_cachelines.csv']
    
    # Process each file
    for path, output_file in zip(file_paths, output_files):
        results = []
        results.extend(parse_file(path, cache_line_size))

        df = pd.DataFrame(results, columns=['CacheLine'])
        df.to_csv(output_file, index=False)
  

if __name__ == "__main__":
    main()
