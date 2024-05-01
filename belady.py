import pandas as pd

def beladys_min(cache_size, access_sequence):
    cache = []
    hits = 0
    for i, access in enumerate(access_sequence):
        if access in cache:
            hits += 1 
        else:
            if len(cache) < cache_size:
                cache.append(access)  
            else:
                future_uses = {item: float('inf') for item in cache}
                for future_index in range(i+1, len(access_sequence)):
                    future_item = access_sequence[future_index]
                    if future_item in future_uses and future_uses[future_item] == float('inf'):
                        future_uses[future_item] = future_index
                
                farthest_item = max(future_uses, key=future_uses.get)
                cache.remove(farthest_item)  
                cache.append(access) 

    return hits / len(access_sequence)  # Return the hit rate

# Example usage:
access_sequence = pd.read_csv('libquantum_cachelines.csv')['CacheLine'].tolist()
# get first 100000
access_sequence = access_sequence[:100000]
cache_size = 10
hit_rate = beladys_min(cache_size, access_sequence)
print(f"Cache hit rate with Belady's MIN algorithm: {hit_rate:.2%}")
