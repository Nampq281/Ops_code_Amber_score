import pandas as pd
import numpy as np
import pickle

def score_scaling(offset, factor, event_p):
    ln_odds = np.log((1-event_p)/event_p)
    score = offset + factor*ln_odds
    score = np.round(score,0)
    return score

