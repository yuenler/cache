from collections import OrderedDict, deque, Counter
import numpy as np
import tensorflow as tf
import random
import pandas as pd
from openai import OpenAI

class Cache:
    def __init__(self, capacity: int, eviction_policy):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.eviction_policy = eviction_policy(self)
        self.access_history = deque(maxlen=50) 
        self.state_snapshots = []
        self.frequency = Counter()
        self.victim_cache_lru = VictimCache(10)
        self.victim_cache_lfu = VictimCache(10)
        self.weights = {'LRU': 0.5, 'LFU': 0.5}

    def adjust_weights(self, policy_hit, key):
        lam = 0.45
        d = 0.005 ** (1/10)
        r = d
        # Adjust the weights based on the policy that hit
        # we want to decrease the weight of the policy that hit and increase the weight of the other policy
        if policy_hit == 'LFU':
            self.weights['LRU'] *= np.exp(-lam * r)
        else:
            self.weights['LFU'] *= np.exp(-lam * r)
        self.weights['LRU'] = self.weights['LRU'] / (self.weights['LRU'] + self.weights['LFU'])
        self.weights['LFU'] = 1 - self.weights['LRU']


    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            evicted_key = self.eviction_policy.evict()
            if evicted_key is not None:
                self.cache.pop(evicted_key, None)
                del self.frequency[evicted_key]
                # print('cache', len(self.cache))
                # print(len(self.frequency))

        # print(key, value)
        self.cache[key] = value

    def access(self, key: int, value: int = 0):
        self.access_history.append(key)
        self.collect_features()
        if self.get(key) == -1:  # cache miss
            # check the victim caches
            if self.victim_cache_lru.access(key) or self.victim_cache_lfu.access(key):
                self.adjust_weights('LRU' if self.victim_cache_lru.access(key) else 'LFU', key)
            self.put(key, value)
            self.frequency[key] = 1
            return False  # Miss
        self.frequency[key] += 1
        return True  # Hit

    def collect_features(self):
        # Take a snapshot of the current state of the cache
        current_state = list(self.cache.keys()) + [-1] * (self.capacity - len(self.cache))
        # Get the last 50 accesses, pad with -1 if fewer than 50
        history_length = 50
        recent_accesses = list(self.access_history)[-history_length:]
        if len(recent_accesses) < history_length:
            recent_accesses += [-1] * (history_length - len(recent_accesses))

        features = {
            "current_cache_state": current_state,
            "recent_access_history": recent_accesses

        }
        return features


class EvictionPolicy:
    def __init__(self, cache):
        self.cache = cache
        self.model = tf.keras.models.load_model('best_model.keras')

    def evict(self):
        raise NotImplementedError


class RandomEvictionPolicy(EvictionPolicy):
    def evict(self):
        return random.choice(list(self.cache.cache.keys()))
        # print(list(self.cache.cache.keys()))
        # return list(self.cache.cache.keys())[0]


class LRUEvictionPolicy(EvictionPolicy):
    def evict(self):
        return next(iter(self.cache.cache))
    
class NNEvictionPolicy(EvictionPolicy):
    def evict(self):
        features = self.cache.collect_features()

        # Convert feature dictionary to a flat array for the NN input
        feature_input = np.array(features['current_cache_state'] + features['recent_access_history'])

        # map each element of the cache to a random number
        unique_numbers = set(features['current_cache_state'] + features['recent_access_history'])
        mapped_values = random.sample(range(1, 101), len(unique_numbers))
        number_map = dict(zip(unique_numbers, mapped_values))

        mapped_cache_order = [number_map[num] for num in features['current_cache_state']]
        mapped_recent_accesses = [number_map[num] for num in features['recent_access_history']]

        # shuffle the cache order
        shuffled_cache_order = mapped_cache_order[:]
        random.shuffle(shuffled_cache_order)

        feature_input = np.array(shuffled_cache_order + mapped_recent_accesses)


        # Predict the key to evict based on the model
        shuffle_predicted_evict_index = np.argmax(self.model.predict(feature_input.reshape(1, -1), verbose=0))
        # print(shuffle_predicted_evict_index)
        # predicted_evict_key = list(self.cache.cache.keys())[predicted_evict_index]
        shuffled_predicted_evict_key = shuffled_cache_order[shuffle_predicted_evict_index]
        predicted_evict_index = mapped_cache_order.index(shuffled_predicted_evict_key)
        predicted_evict_key = list(self.cache.cache.keys())[predicted_evict_index]
        # print(shuffle_predicted_evict_index)
        # print(predicted_evict_key)
        return predicted_evict_key


class PredictiveEvictionPolicy(EvictionPolicy):
    def predict_accesses(self, current_accesses):
        # pattern_key = tuple(current_accesses)
        # if pattern_key in self.history_pattern_cache:
        #     return self.history_pattern_cache[pattern_key]
        client = OpenAI()

        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Continue the sequence of numbers given to you. Separate each number with a comma. Return 50 numbers."},
                    {"role": "user", "content": " ".join(map(str, current_accesses))}
                ]
                )
            print(completion.choices[0].message.content)
            predicted_accesses = completion.choices[0].message.content.split(',')
            predicted_accesses = [int(access.strip()) for access in predicted_accesses]
        except Exception as e:
            print(f"Failed to predict with GPT-3.5: {e}")
            predicted_accesses = []

        # self.history_pattern_cache[pattern_key] = predicted_accesses
        # print(predicted_accesses)
        return predicted_accesses

    def evict(self):
        current_accesses = list(self.cache.access_history)
        predicted_accesses = self.predict_accesses(current_accesses)

        future_demand = {}
        for idx, access in enumerate(predicted_accesses):
            if access not in future_demand:
                future_demand[access] = idx

        # Apply Belady's optimal algorithm
        max_index = -1
        evict_key = None
        for key in self.cache.cache.keys():
            if key in future_demand:
                if future_demand[key] > max_index:
                    max_index = future_demand[key]
                    evict_key = key
            else:
                return key 

        return evict_key if evict_key is not None else random.choice(list(self.cache.cache.keys()))


class LFUEvictionPolicy(EvictionPolicy):
    def evict(self):
        if not self.cache.frequency:
            print('ERROR: Frequency dictionary is empty, no items to evict.')
            return None

        least_freq = min(self.cache.frequency.values())
        potential_keys = [k for k, v in self.cache.frequency.items() if v == least_freq]
        if not potential_keys:
            print('ERROR: No keys found with the minimum frequency for eviction.')
            return None

        # print(potential_keys[0])
        # return potential_keys[0]
        # print(potential_keys[0], potential_keys[0] in self.cache.cache.kßeys())
        return random.choice(list(self.cache.cache.keys()))


class HybridEvictionPolicy(EvictionPolicy):
    def __init__(self, cache):
        super().__init__(cache)
        self.lru = LRUEvictionPolicy(cache)
        self.lfu = LFUEvictionPolicy(cache)

    def evict(self):

        # print(self.cache.cache)
        # print(self.cache.frequency)

        # print(self.cache.weights['LRU'], self.cache.weights['LFU'])

        if random.random() < self.cache.weights['LRU']:
            evicted_key = self.lru.evict()
            self.cache.victim_cache_lru.add(evicted_key, self.cache.cache[evicted_key])

        else:
            evicted_key = self.lfu.evict()
            self.cache.victim_cache_lfu.add(evicted_key, self.cache.cache[evicted_key])

        return evicted_key




class VictimCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def access(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return True
        return False

    def add(self, key, value):
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def remove(self, key):
        if key in self.cache:
            del self.cache[key]





cache_size = 10
random_cache = Cache(cache_size, RandomEvictionPolicy)
lru_cache = Cache(cache_size, LRUEvictionPolicy)
lfu_cache = Cache(cache_size, LFUEvictionPolicy)
nn_cache = Cache(cache_size, NNEvictionPolicy)
predictive_cache = Cache(cache_size, PredictiveEvictionPolicy)
hybrid_cache = Cache(cache_size, HybridEvictionPolicy)



access_sequence = pd.read_csv('hmmer_cachelines.csv')['CacheLine'].tolist()

access_sequence = access_sequence[:300000]




random_hits = 0
lru_hits = 0
nn_hits = 0
predictive_hits = 0
hybrid_hits = 0
lfu_hits = 0


for key in access_sequence:
    if random_cache.access(key, value=key):
        random_hits += 1

    if lru_cache.access(key, value=key):
        lru_hits += 1
    if lfu_cache.access(key, value=key):
        lfu_hits += 1
    # # else:
    # #     print(f"Miss for key: {key}")
    # if nn_cache.access(key, value=key):
    #     nn_hits += 1

    # if predictive_cache.access(key, value=key):
    #     predictive_hits += 1

    if hybrid_cache.access(key, value=key):
        hybrid_hits += 1


    # else:
    #     print(f"Miss for key: {key}")

print(f"Random Cache Hits: {random_hits}")
print(f"LRU Cache Hits: {lru_hits}")
print(f"LFU Cache Hits: {lfu_hits}")
print(f"NN Cache Hits: {nn_hits}")
print(f"Predictive Cache Hits: {predictive_hits}")
print(f"Hybrid Cache Hits: {hybrid_hits}")
# print("Feature Data Collected:")
# for f in cache.state_snapshots:
#     print(f)
