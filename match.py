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
17: First, let's get a few things out of the way
What grade are you in?
18: What sport do you do?
19: Major? (or interest if no major yet)
20: Do you want to be placed in a group with one teammate or friend? You should both put each other's name for this question!
21: Do you want to be placed with people in your grade?
22: Would you rather do a 1-on-1?
23: Would you be interested in a student-athlete speed-dating event in the future?

############### Questions ###############
24-52: Questions
"""

import csv
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sb

def loadData(filename):
    print(f"{'=' * 10} Loading data {'=' * 10}")
    
    people = []
    
    with open(filename) as f:
        question_ids, questions, _, *responses = csv.reader(f)
        
        for person_id, row in enumerate(responses):
            person = {
                'id': person_id,
                'start date': row[0],
                'end date': row[1],
                'qualtrics_id': row[8],
                'meta data': {
                    'year': row[17],
                    'sport': row[18],
                    'major': row[19],
                },
                'athlete mingle': {
                    'friend': row[20],
                    'same grade': row[21],
                    '1-on-1': row[22],
                    'speed-dating': row[23],
                },
                'responses': [int(x) if x != '' else 0 for x in row[24:]]
            }
            people.append(person)
            
    print(f'There were {len(people)} responses')
    # print(json.dumps(people[-1], indent=4))
    
    return people

def cleanData(people):
    print(f"\n{'=' * 10} Cleaning data {'=' * 10}")
    
    return people

def analyzeData(people):
    print(f"\n{'=' * 10} Analyzing data {'=' * 10}")
    
    responses = np.array([person['responses'] for person in people])
    print(responses)
    
    fig, axs = plt.subplots(6, 5)
    fig.set_size_inches(18.5, 10.5)
    plt.subplots_adjust(left=0.05, right=0.95, hspace=0.6)
    
    for q in range(len(responses[0])):
        # TODO: add percentage labels for each bin (plt.annotate)
        axs[q // 5, q % 5].hist(responses[:, q], bins=5)
        axs[q // 5, q % 5].set_title(f"Question {q}")
        axs[q // 5, q % 5].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[q // 5, q % 5].yaxis.set_major_locator(MaxNLocator(integer=True))
        
    plt.savefig('output/histograms.png')

def computeSimilarity(person1, person2):
    """
    Returns the percentage of questions that were answered the same.
    """
    # check if person1 and person2 are the same person
    if person1['qualtrics_id'] == person2['qualtrics_id']: return 0
    
    responses1 = np.array(person1['responses'])
    responses2 = np.array(person2['responses'])
    
    return np.count_nonzero(responses1 == responses2) / len(responses1)

def match(people):
    print(f"\n{'=' * 10} Making matches {'=' * 10}")
    
    # compute compatibilities
    scores = np.zeros((len(people), len(people)))
    for i, person1 in enumerate(people):
        for j, person2 in enumerate(people):
            scores[i][j] = computeSimilarity(person1, person2)
    
    print(scores)
    print()
    fig, ax = plt.subplots(figsize=(11, 9))
    sb.heatmap(scores, annot=True)
    plt.savefig('output/confusion_matrix.png')
    
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
    people = loadData('data/Athlete Mingle_February 10, 2021_21.20.csv')
    people = cleanData(people)
    analyzeData(people)
    matches = match(people)

if __name__ == '__main__':
    main()