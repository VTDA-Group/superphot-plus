# Prepares the JSON file for custom benchmarks.
#
# More information: https://github.com/benchmark-action/github-action-benchmark/pull/81/files
# [{
#   "name": "My Custom Smaller Is Better Benchmark - Memory Used",
#   "unit": "Megabytes",
#   "value": 100,
#   "range": "3",
#   "extra": "Value for Tooltip: 25\nOptional Num #2: 100\nAnything Else!",
# }]

import json

benchmarks = []

with open("benchmarks/results.json", "r+") as results_json:
    # Read results from the pytest-monitor output
    results = json.load(results_json)

    for result in results:
        total_time = {
            "name": "Runtime",
            "unit": "Seconds",
            "value": result["TOTAL_TIME"]
        }
        mem_usage = {
            "name": "Memory Used",
            "unit": "Megabytes",
            "value": result["MEM_USAGE"]
        }
        benchmarks.extend([total_time, mem_usage])

    # Write the formatted results on a clean slate
    results_json.truncate()
    results_json.seek(0)
    json.dump(benchmarks, results_json)
