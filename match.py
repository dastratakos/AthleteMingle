"""
file: match.py
date: 2/10/21
author: Dean Stratakos
----------------------

Each row has the following columns:

############### Meta-Data ###############
0: Start Date
1: End Date
2: Response Type
3: IP Address
4: Progress
5: Duration (in seconds)
6: Finished
7: Recorded Date
8: Response ID
9: Recipient Last Name
10: Recipient First Name
11: Recipient Email
12: External Data Reference
13: Location Latitude
14: Location Longitude
15: Distribution Channel
16: User Language

############### Athlete Mingle Data ###############
17: First and last name
18: Stanford email
19: What year are you?
20: What sport do you do?
21: Major? (or interest if no major yet)
22: Would you rather be matched 1-on-1 or in a small group?
23: If you want one teammate or friend to be in your group, you should both put
    each other's name for this question! Leave this question blank if you'd
    like your entire group to be a ~surprise~.
24: Do you want to be placed with people in your year?

25-53

54: Unrelated to Athlete Mingle: would you be interested in a student-athlete
    speed dating event in the future?

############### Extras ###############
55: Q7 - Parent Topics
56: Q7 - Sentiment Polarity
57: Q7 - Sentiment Score
58: Q7 - Sentiment
59: Q7 - Topic Sentiment Label
60: Q7 - Topic Sentiment Score
61: Q7 - Topics
"""

import csv
import json
import os

# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
import numpy as np
# import seaborn as sb

from plot import *

TEXT_FILE = 'data/Athlete Mingle_February 18, 2021_11.25.csv'
NUMBER_FILE = 'data/Athlete Mingle_February 18, 2021_11.24.csv'
OUT_PATH = 'output/02-19/'

SPORTS = {}
with open('sports.json') as f:
    SPORTS = json.load(f)

print(SPORTS)

# TODO: TAKE OUT KRISTEN

def loadData():
    print(f"{'=' * 10} Loading data {'=' * 10}")
        
    people = []
    skip_ids = []
    
    with open(NUMBER_FILE) as f:
        _, column_headers, _, *responses = csv.reader(f)
        
        for row in responses:
            for res in row[25:54]:
                if res == '':
                    skip_ids.append(row[8])
                
        # for i, header in enumerate(column_headers):
        #     print(f"{i}: {header}")
        i = 0
        for row in responses:
            if row[8] in skip_ids:
                continue
            
            person = {
                'index': i,
                'qualtrics_id': row[8],
                'start date': row[0],
                'end date': row[1],
                'meta data': {
                    'name': row[17].strip(),
                    'email': row[18],
                    'sport_id': int(row[20]),
                },
                'athlete mingle': {
                    '1-on-1': row[22] == '1',
                    'friend': row[23],
                    'same grade': row[24] == '1',
                },
                'responses': [int(x) for x in row[25:54]],
                'speed-dating': row[54] == '1',
            }
            people.append(person)
            
            i += 1
            
    with open(TEXT_FILE) as f:
        _, column_headers, _, *responses = csv.reader(f)
                
        # for i, header in enumerate(column_headers):
        #     print(f"{i}: {header}")
        
        i = 0
        for row in responses:
            if row[8] in skip_ids:
                continue
            
            assert people[i]['qualtrics_id'] == row[8]
            
            people[i]['meta_data']['year'] = row[19]
            people[i]['meta_data']['sport_name'] = row[20]
            people[i]['meta_data']['major'] = row[21]
            
            i += 1
            
    print(f'There were {len(people)} responses ' + \
          f'and {len(skip_ids)} unfinished surveys.')
    print(json.dumps(people[-1], indent=4))
    
    return people

# TODO: TAKE OUT KRISTEN

def analyzeData(people):
    print(f"\n{'=' * 10} Analyzing data {'=' * 10}")
    
    year_labels = ['Frosh', 'Sophomore', 'Junior', 'Senior', '5th Year/Coterm']
    sport_labels = [sport['sport'] for sport in SPORTS.values()]
    gender_labels = ['Men', 'Women', 'Co-Ed']
    
    
    ########## Analyze years ##########
    
    years = [person['meta data']['year'] for person in people]
    counts = [years.count(i) for i in range(1, len(year_labels) + 1)]
    
    plotYears(year_labels, counts, out_path=f'{OUT_PATH}/years.png')
        
    ########## Analyze sports ##########
    
    res_sports = [person['meta data']['sport_name'] for person in people]
    counts = [res_sports.count(label) for label in sport_labels]
    
    plotSports(sport_labels, counts, out_path=f'{OUT_PATH}sports.png')
    
    ########## Analyze genders ##########
    
    counts = [0, 0, 0]
    for res in res_sports:
        for sport in SPORTS.values():
            if sport['sport'] == res:
                counts[gender_labels.index(sport['gender'])] += 1
    
    plotGenders(gender_labels, counts, out_path=f'{OUT_PATH}genders.png')
    
    ########## Analyze year/gender ##########
    
    counts = np.array((3, len(year_labels)))
    
    for y, s in zip(years, res_sports):
        gender = None
        for sport in SPORTS.values():
            if sport['sport'] == s:
                gender = sport['gender']
                break
            
        counts[gender_labels.index(gender)][y - 1] += 1

    plotYearGender(year_labels, counts, out_path=f'{OUT_PATH}years-genders.png')
    
    ########## Analyze response distribution ##########
    
    responses = np.array([person['responses'] for person in people])
    
    plotResponses(responses, out_path=f'{OUT_PATH}responses.png')

# TODO: TAKE OUT KRISTEN

def computeSimilarity(person1, person2):
    """
    Returns the percentage of questions that were answered the same.
    """
    # check if person1 and person2 are the same person
    if person1['qualtrics_id'] == person2['qualtrics_id']: return 0
    
    responses1 = np.array(person1['responses'])
    responses2 = np.array(person2['responses'])
    
    return np.count_nonzero(responses1 == responses2) / len(responses1)

# TODO: TAKE OUT KRISTEN

def match(people):
    print(f"\n{'=' * 10} Making matches {'=' * 10}")
    
    # compute compatibilities
    scores = np.zeros((len(people), len(people)))
    for i, person1 in enumerate(people):
        for j, person2 in enumerate(people):
            scores[i][j] = computeSimilarity(person1, person2)
    
    print(scores)
    print()
    plotHeatMap(scores, out_path=f'{OUT_PATH}confusion_matrix.png')
    
    # select best overall pairs
    """
    Greedy algorithm: find the highest match, then pair those two and remove
    from set of matches.
    """
    unmatched = set(range(len(people)))
    matches = []
    while (len(unmatched) > 1):
        ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        matches.append([ind, scores[ind]])
        
        for i in ind:
            scores[i, :] = -1
            scores[:, i] = -1
            unmatched.remove(i)
        
    community_score = 0.0
    for match in matches:
        print(f"Person {match[0][0]} matched with Person {match[0][1]} ({match[1]:.4%})")
        community_score += match[1]
    community_score /= len(matches)
    
    print(f"\nCommunity score: {community_score:.4%}")
    
    return matches

def main():
    people = loadData()
    # analyzeData(people)
    # matches = match(people)

# TODO: TAKE OUT KRISTEN

if __name__ == '__main__':
    main()
    
# TODO: TAKE OUT KRISTEN