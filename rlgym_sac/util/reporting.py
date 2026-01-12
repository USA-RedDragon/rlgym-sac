"""
File: reporting.py
Author: Matthew Allen

Description:
    Misc. functions to log a dictionary of metrics to wandb and print them to the console.
"""


import torch
import numpy as np
import locale
try:
    locale.setlocale(locale.LC_ALL, '')
except locale.Error:
    pass # fallback


def _form_printable_groups(report):
    """
    Function to create a list of dictionaries containing the data to print to the console in a specific order.
    :param report: A dictionary containing all of the values to organize.
    :return: A list of dictionaries containing keys organized in the desired fashion.
    """
    
    # helper to safely get key
    def safe_get(key):
        return report.get(key, "N/A")

    groups = [
        {"Policy Reward": safe_get("Policy Reward"),
         # SAC/PPO common
         "Value Function Loss": safe_get("Value Function Loss"),
         "Policy Loss": safe_get("Policy Loss"),
         "Alpha": safe_get("Alpha"),
         "Alpha Loss": safe_get("Alpha Loss"),
         "Mean Q Value": safe_get("Mean Q Value")},
        
        {"Collected Steps per Second": safe_get("Collected Steps per Second"),
         "Overall Steps per Second": safe_get("Overall Steps per Second")},

        {"Timestep Collection Time": safe_get("Timestep Collection Time"),
         "Timestep Consumption Time": safe_get("Timestep Consumption Time"),
         "Total Iteration Time": safe_get("Total Iteration Time")},

        {"Cumulative Model Updates": safe_get("Cumulative Model Updates"),
         "Cumulative Timesteps": safe_get("Cumulative Timesteps")},

        {"Timesteps Collected": safe_get("Timesteps Collected")},
    ]

    # Add other keys that are in report but not in groups
    covered_keys = set()
    for g in groups:
        covered_keys.update(g.keys())
    
    rest = {k: v for k, v in report.items() if k not in covered_keys}
    if rest:
        groups.append(rest)

    return groups

def report_metrics(loggable_metrics, debug_metrics, wandb_run=None):
    """
    Function to report a dictionary of metrics to the console and wandb.
    :param loggable_metrics: Dictionary containing all the data to be logged.
    :param debug_metrics: Optional dictionary containing extra data to be printed to the console for debugging.
    :param wandb_run: Wandb run to log to.
    :return: None.
    """

    if wandb_run is not None:
        wandb_run.log(loggable_metrics)

    # Print debug data first.
    if debug_metrics is not None:
        print("\nBEGIN DEBUG\n")
        print(dump_dict_to_debug_string(debug_metrics))
        print("\nEND DEBUG\n")


    # Print the loggable metrics in a desirable format to the console.
    print("{}{}{}".format("-"*8, "BEGIN ITERATION REPORT", "-"*8))
    groups = _form_printable_groups(loggable_metrics)
    out = ""
    for group in groups:
        # Filter out N/A for clean printing
        group = {k: v for k, v in group.items() if v != "N/A"}
        if group:
            out += dump_dict_to_debug_string(group) + "\n"
    print(out[:-2])
    print("{}{}{}\n\n".format("-"*8, "END ITERATION REPORT", "-"*8))

def dump_dict_to_debug_string(dictionary):
    """
    Function to format the data in a loggable dictionary so the line length is limited.

    :param dictionary: Data to format.
    :return: A string containing the formatted elements of that dictionary.
    """

    debug_string = ""
    for key, val in dictionary.items():
        if isinstance(val, torch.Tensor):
            if len(val.shape) == 0:
                val = val.detach().cpu().item()
            else:
                val = val.detach().cpu().tolist()

        # Format lists of numbers as [num_1, num_2, num_3] where num_n is clipped at 5 decimal places.
        if isinstance(val, (tuple, list, np.ndarray)):
            arr_str = []
            for arg in val:
                if isinstance(arg, float):
                    arr_str.append(f"{arg:7.5f}")
                else:
                    arr_str.append(f"{arg},")

            arr_str = ' '.join(arr_str)
            debug_string = "{}{}: [{}]\n".format(debug_string, key, arr_str[:-1])

        # Format floats such that only 5 decimal places are shown.
        elif isinstance(val, (float, np.float32, np.float64)):
            debug_string = "{}{}: {:7.5f}\n".format(debug_string, key, val)

        # Print ints with comma separated thousands (locale aware).
        elif isinstance(val, (int, np.int32, np.int64)):
             # Use f-string for locale aware printing if possible, or basic
            try:
                debug_string = "{}{}: {:n}\n".format(debug_string, key, val)
            except ValueError:
                debug_string = "{}{}: {}\n".format(debug_string, key, val)
        
        # Default to just printing the value if it isn't a type we know how to format.
        else:
            debug_string = "{}{}: {}\n".format(debug_string, key, val)

    return debug_string
