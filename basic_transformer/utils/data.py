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


def generate_fake_df():
    """
    Generate fake df for debugging
    """
    n_positive, n_negative = 5000, 5000
    positive_label, negative_label = 'positive', 'negative'
    positive_text = ' '.join(['good'] * 10)
    negative_text = ' '.join(['bad'] * 10)
    df = [(positive_text, positive_label)] * n_positive + [(negative_text, negative_label)] * n_negative
    df = pd.DataFrame(df)
    df.columns = ('review', 'sentiment')
    return df
