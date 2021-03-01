import json
import random

import numpy as np

SPORT_DATA = {}
with open('data/sports.json') as f:
    SPORT_DATA = json.load(f)

def computeSimilarity(p1, p2, same_kind_ok=False, same_sport_ok=False):
    """
    Returns the similarity between two people. The raw score is the percentage
    of questions that were answered the same. There are additional factors
    taken into account:
        - do not match someone with themselves.
        - respect grade preferences
        - avoid matching people on the same team
        - avoid matching people on counterpart teams (e.g. Men's Basketball and
          Women's Basketball)
        - avoid matching people with a grade difference of more than 2 years
    """
    # check if p1 and p2 are the same person
    if p1['qualtrics id'] == p2['qualtrics id']: return -1.5
    
    sport1 = p1['meta data']['sport name']
    sport2 = p2['meta data']['sport name']

    penalties = 0

    if p1['athlete mingle']['same grade'] or p2['athlete mingle']['same grade']:
        if p1['meta data']['year'] != p2['meta data']['year']:
            penalties += 0.5

    # check if they are on the same team
    if not same_sport_ok and sport1 == sport2:
        penalties += 0.3
        
    # check if they are on counterpart teams
    counterpart1 = SPORT_DATA[sport1].get('counterpart', 'none1')
    counterpart2 = SPORT_DATA[sport2].get('counterpart', 'none2')
    if (counterpart1 == sport2 or counterpart2 == sport1):
        penalties += 0.1
    
    if penalties > 0:
        return -penalties
    
    responses1 = np.array(p1['responses'])
    responses2 = np.array(p2['responses'])
    
    raw_score = np.count_nonzero(responses1 == responses2) / len(responses1)
    
    # minimize far away grades
    year_labels = ['Frosh', 'Sophomore', 'Junior', 'Senior', '5th year/Coterm']
    if abs(year_labels.index(p1['meta data']['year']) -
           year_labels.index(p2['meta data']['year'])) > 2:
        raw_score *= 0.1
    
    return raw_score