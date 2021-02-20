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

import numpy as np
from tqdm import tqdm

from plot import *

TEXT_FILE = 'data/Athlete Mingle_February 18, 2021_11.25.csv'
NUMBER_FILE = 'data/Athlete Mingle_February 18, 2021_11.24.csv'
OUT_PATH = 'output/02-19/'

SPORT_DATA = {}
with open('sports.json') as f:
    SPORT_DATA = json.load(f)

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
                    break
        
        i = 0
        for row in responses:
            if row[8] in skip_ids:
                continue
            
            person = {
                'index': i,
                'qualtrics id': row[8],
                'start date': row[0],
                'end date': row[1],
                'meta data': {
                    'name': row[17].strip(),
                    'email': row[18],
                    'sport id': int(row[20]),
                },
                'athlete mingle': {
                    '1-on-1': row[22] == '1',
                    'friend': row[23],
                    'same grade': row[24] == '1',
                    'speed-dating': row[54] == '1',
                },
                'responses': [int(x) for x in row[25:54]],
            }
            people.append(person)
            
            i += 1
            
    with open(TEXT_FILE) as f:
        _, column_headers, _, *responses = csv.reader(f)
        
        i = 0
        for row in responses:
            if row[8] in skip_ids:
                continue
            
            assert people[i]['qualtrics id'] == row[8]
            
            people[i]['meta data']['year'] = row[19]
            people[i]['meta data']['sport name'] = row[20]
            people[i]['meta data']['major'] = row[21]
            
            i += 1
            
    print(f'There were {len(people)} responses ' + \
          f'and {len(skip_ids)} unfinished surveys.')
    # print(json.dumps(people[-1], indent=4))
    
    return people

# TODO: TAKE OUT KRISTEN

def analyzeData(people):
    print(f"\n{'=' * 10} Analyzing data {'=' * 10}")
    
    year_labels = ['Frosh', 'Sophomore', 'Junior', 'Senior', '5th year/Coterm']
    major_labels = [
        'Aeronautics and Astronautics',
        'Biology',
        'Chemistry',
        'Computer Science',
        'Earth Systems',
        'Econ',
        'Engineering',
        'English',
        'History',
        'Human Biology',
        'MS&E',
        'Physics',
        'Political Science',
        'Product Design',
        'Psychology',
        'STS',
        'SymSys',
        'Other'
    ]
    sport_labels = [sport['sport'] for sport in SPORT_DATA.values()]
    gender_labels = ['Men', 'Women', 'Co-Ed']
    
    
    ########## Analyze years ##########
    
    years = [person['meta data']['year'] for person in people]
    counts = [years.count(y) for y in year_labels]
    
    plotYears(year_labels, counts, out_path=f'{OUT_PATH}/years.png')
        
    ########## Analyze majors ##########
    
    majors = [person['meta data']['major'] for person in people]
    counts = [majors.count(label) for label in major_labels]
    
    plotMajors(major_labels, counts, out_path=f'{OUT_PATH}majors.png')
    
    ########## Analyze sports ##########
    
    sports = [person['meta data']['sport name'] for person in people]
    counts = [sports.count(label) for label in sport_labels]
    
    plotSports(sport_labels, counts, out_path=f'{OUT_PATH}sports.png')
    
    ########## Analyze genders ##########
    
    counts = [0, 0, 0]
    for res in sports:
        for sport in SPORT_DATA.values():
            if sport['sport'] == res:
                counts[gender_labels.index(sport['gender'])] += 1
    
    plotGenders(gender_labels, counts, out_path=f'{OUT_PATH}genders.png')
    
    ########## Analyze year/gender ##########
    
    counts = np.zeros((3, len(year_labels)), dtype='intc')
    
    for y, s in zip(years, sports):
        gender = None
        for sport in SPORT_DATA.values():
            if sport['sport'] == s:
                gender = sport['gender']
                break
            
        counts[gender_labels.index(gender)][year_labels.index(y)] += 1

    plotYearGender(year_labels, counts, out_path=f'{OUT_PATH}yearsGenders.png')
    
    ########## Analyze response distribution ##########
    
    responses = np.array([person['responses'] for person in people])
    
    plotResponses(responses, out_path=f'{OUT_PATH}responses.png')

# TODO: TAKE OUT KRISTEN

def computeSimilarity(person1, person2):
    """
    Returns the percentage of questions that were answered the same.
    """
    # check if person1 and person2 are the same person
    if person1['qualtrics id'] == person2['qualtrics id']: return 0
    
    responses1 = np.array(person1['responses'])
    responses2 = np.array(person2['responses'])
    
    return np.count_nonzero(responses1 == responses2) / len(responses1)

# TODO: TAKE OUT KRISTEN

def match(people):
    print(f"\n{'=' * 10} Making matches {'=' * 10}")
    
    # compute compatibilities
    print('Computing compatibilities')
    scores = np.zeros((len(people), len(people)))
    with tqdm(total=len(people)) as progress_bar:
        for i, person1 in enumerate(people):
            progress_bar.update()
            for j, person2 in enumerate(people):
                scores[i][j] = computeSimilarity(person1, person2)
    
    print(scores)
    # print('Plotting heat map')
    # plotHeatMap(scores, out_path=f'{OUT_PATH}confusion_matrix.png')
    print()
    
    # select best overall pairs
    """
    Greedy algorithm: find the highest match, then pair those two and remove
    from set of matches.
    """
    print('Creating matches')
    unmatched = set(range(len(people)))
    matches = []
    with tqdm(total=len(unmatched)) as progress_bar:
        while (len(unmatched) > 1):
            ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
            matches.append([ind, scores[ind]])
            
            for i in ind:
                scores[i, :] = -1
                scores[:, i] = -1
                unmatched.remove(i)
                progress_bar.update()
    
    with open(f'{OUT_PATH}matches.csv', 'w') as f:
        f.write('p1 id,p1 name,p1 email,p1 sport,' + \
                'p2 id,p2 name,p2 email,p2 sport,' + \
                'score')
        
        community_score = 0.0
        
        for match in matches:
            p1, p2 = match[0]
            p1_info = people[p1]['meta data']
            p2_info = people[p2]['meta data']
            
            print(f"Person {p1} matched with Person {p2} ({match[1]:.4%})")
            f.write(f"{p1},{p1_info['name']},{p1_info['email']}," + \
                    f"{p1_info['sport name']}," + \
                    f"{p2},{p2_info['name']},{p2_info['email']}," + \
                    f"{p2_info['sport name']}," + \
                    f"{match[1]:.4%}\n")
            
            community_score += match[1]
            
        community_score /= len(matches)
        
        message = f"\nCommunity score: {community_score:.4%}"
        print(message)
        f.write(message)
    
    return matches

def main():
    people = loadData()
    analyzeData(people)
    matches = match(people)

# TODO: TAKE OUT KRISTEN

if __name__ == '__main__':
    main()
    
# TODO: TAKE OUT KRISTEN