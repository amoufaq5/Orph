# orphtools/preprocessing/dataset_merger.py

import json
import os
from pathlib import Path
from datetime import datetime

def merge_datasets(sources, output_dir):
    merged = []
    for path in sources:
        with open(path, "r") as f:
            merged += json.load(f)

    # Remove exact duplicates
    unique_data = {json.dumps(entry, sort_keys=True) for entry in merged}
    final_data = [json.loads(item) for item in unique_data]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"merged_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"✅ Merged {len(sources)} datasets. Saved to {output_path}")
    return output_path
