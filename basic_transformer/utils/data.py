import numpy as np


def fix_seq_len(s, max_len, fill_token=0):
    
    # --- input check --
    if type(s) is not list:
        raise ValueError("Input must be of `list` type.")
        
    if len(s) == max_len:
        pass
    elif len(s) > max_len:
        s = s[:max_len]
    else:
        s = s + [fill_token] * (max_len - len(s))
        
    return s