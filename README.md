# ML for Cache Replacement

To reproduce the results, follow the steps below:

1. Run `get_lines.py` to convert the list of addresses to block numbers
2. Run `dataset_generation.py` to generate the dataset for Approach #1
3. Train the model for Approach #1 by running `model.py`.
4. Create a `.env` file with the following content:
```
OPENAI_API_KEY=<your_openai_api_key>
```
5. Finally, run `cache.py` to compare all cache replacement algorithms.
