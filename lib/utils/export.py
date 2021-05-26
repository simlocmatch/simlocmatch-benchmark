"""Exporting for submission.
"""
from typing import Dict
import flammkuchen as dd

def to_h5(results: Dict, task: str, method_details: Dict, args: Dict):
    """Export results in a single .h5 file for submission. 
    """
    output = {"method_results": results, "task": task,
              "method_details": method_details, "args":args}
    dd.save(f"{task}-{method_details['MethodTitle']}.h5", output,
            compression="blosc")
