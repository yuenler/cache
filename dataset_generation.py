import pandas as pd
from collections import defaultdict, deque
import numpy as np
import random

def beladys_min_random_evict(access_sequence, cache_size):
    future_accesses = defaultdict(deque)
    for i, v in enumerate(access_sequence):
        future_accesses[v].append(i)

    cache = set()
    cache_order = deque()
    recent_accesses = deque(maxlen=50)
    data = []

    for i, v in enumerate(access_sequence):
        recent_accesses.append(v)

        future_accesses[v].popleft()
        
        if v in cache:
            cache_order.remove(v)
            cache_order.append(v)
            continue

        if len(cache) < cache_size:
            cache.add(v)
            cache_order.append(v)
        else:
            farthest_access = -1
            evict_candidate = None
            for item in cache_order:
                if future_accesses[item]:
                    next_access = future_accesses[item][0]
                else:
                    next_access = float('inf') 
                if next_access > farthest_access:
                    farthest_access = next_access
                    evict_candidate = item

            # Determine the label using Belady's MIN but evict a random item
            random_evict_candidate = random.choice(list(cache_order))

            # Mapping cache lines and recent accesses to random values for simulation data
            all_numbers = list(cache_order) + list(recent_accesses)
            unique_numbers = set(all_numbers)
            mapped_values = random.sample(range(1, 101), len(unique_numbers))
            number_map = dict(zip(unique_numbers, mapped_values))

            mapped_cache_order = [number_map[num] for num in cache_order]
            mapped_recent_accesses = [number_map[num] for num in recent_accesses]

            beladys_mapped_evict_candidate = number_map[evict_candidate]

            # shuffle the cache order and change the label to the index of the evicted item
            random.shuffle(mapped_cache_order)

            feature_row = {'cache_line_{}'.format(j+1): mapped_cache_order[j] if j < len(mapped_cache_order) else -1 for j in range(cache_size)}
            for j in range(50):
                feature_row['recent_access_{}'.format(j+1)] = mapped_recent_accesses[j] if j < len(mapped_recent_accesses) else -1
            feature_row['label'] = mapped_cache_order.index(beladys_mapped_evict_candidate)
            
            data.append(feature_row)

            # Evict the randomly selected item and add the new item
            cache.remove(random_evict_candidate)
            cache_order.remove(random_evict_candidate)
            cache.add(v)
            cache_order.append(v)

    return pd.DataFrame(data)

cache_size = 10
access_sequence = pd.read_csv('libquantum_cachelines.csv')['CacheLine'].tolist()

df = beladys_min_random_evict(access_sequence, cache_size)
df.to_csv('cache_simulation_data.csv', index=False)
