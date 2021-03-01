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

############### Compatibility questions ###############
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
import copy
import json
import os
import random

import numpy as np

from similarity import *
from plot import *

TEXT_FILE = 'data/Athlete Mingle_February 18, 2021_11.25.csv'
NUMBER_FILE = 'data/Athlete Mingle_February 18, 2021_11.24.csv'
OUT_PATH = 'output/03-01/'
PLOT_PATH = OUT_PATH + 'plots/'

MATCH_HEADER = 'p1 id,p1 name,p1 email,p1 sport,p1 year,p1 same grade,' + \
               'p2 id,p2 name,p2 email,p2 sport,p2 year,p2 same grade,' + \
               'score,question,response\n'

################################ Loading data #################################

SPORT_DATA = {}
with open('sports.json') as f:
    SPORT_DATA = json.load(f)

def loadData(verbose=False):
    print(f"{'=' * 10} Loading data {'=' * 10}")
        
    people = []
    skip_ids = []
    
    with open(NUMBER_FILE) as f:
        _, column_headers, _, *responses = csv.reader(f)
        
        if verbose:
            for i, column in enumerate(column_headers[25:54]):
                print(f"{i}: {column}")
        
        """ Skip unfinished responses. """
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
                    'name': ' '.join(row[17].strip().split()).title(),
                    'email': row[18].strip().lower(),
                    'sport id': int(row[20]),
                },
                'athlete mingle': {
                    '1-on-1': row[22] == '1',
                    'friends': [f.strip() for f in row[23].split(',')
                                if f != ''],
                    'same grade': row[24] == '1',
                    'speed-dating': row[54] == '1',
                },
                'responses': [int(x) for x in row[25:54]],
            }
            people.append(person)
            
            i += 1
    
    """ Add additional string data to each person. """
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
    
    """ Randomize people so that the timestamp doesn't influence matches. """
    # random.seed(42)
    random.shuffle(people)
            
    print(f'There were {len(people)} responses ' + \
          f'and {len(skip_ids)} unfinished surveys.')
    
    if verbose:
        print(json.dumps(people[-1], indent=4))
    
    return people

################################ Data analysis ################################

def analyzeData(people):
    print(f"\n{'=' * 10} Analyzing data {'=' * 10}")
    
    years = [person['meta data']['year'] for person in people]
    majors = [person['meta data']['major'] for person in people]
    sports = [person['meta data']['sport name'] for person in people]
    responses = np.array([person['responses'] for person in people])

    plotYears(years, out_path=f'{PLOT_PATH}years.png')
    plotMajors(majors, out_path=f'{PLOT_PATH}majors.png')
    plotSports(sports, out_path=f'{PLOT_PATH}sports.png')
    plotKinds(sports, out_path=f'{PLOT_PATH}kinds.png')
    plotYearKind(years, sports, out_path=f'{PLOT_PATH}yearsKinds.png')
    plotResponses(responses, out_path=f'{PLOT_PATH}responses.png')

############################## Helper functions ###############################

def separatePairsAndGroups(people):
    one_on_one = []
    group = []
    for person in people:
        if person['athlete mingle']['1-on-1']:
            one_on_one.append(person)
        else:
            group.append(person)
    return one_on_one, group

def findPerson(target_name, people):
    for person in people:
        if person['meta data']['name'] == target_name:
            return person

def computeSimilarities(people, same_kind_ok=False):
    print('Computing compatibilities')
    
    scores = np.zeros((len(people), len(people)))
    for i, p1 in enumerate(people):
        for j, p2 in enumerate(people):
            scores[i][j] = computeSimilarity(p1, p2, same_kind_ok=same_kind_ok)
    
    print(scores)
    # plotHeatMap(scores, out_path=f'{OUT_PATH}confusion_matrix.png')
    print()
    
    return scores
    
def findSimilarResponse(p1, p2, p3=None, p4=None):
    """
    Returns a random similar response. A response is the same if for each
    person, the person is None, or the response is the same as p1's response.
    
    Here is the numpy code when there are only 2 people:
        same_responses = np.argwhere(res1 == res2).flatten()
    """
    res1 = np.array(p1['responses'])
    res2 = np.array(p2['responses'])
    res3 = np.array(p3['responses']) if p3 else None
    res4 = np.array(p4['responses']) if p4 else None
    
    same_responses = []
    for i in range(len(res1)):
        if ((res1[i] == res2[i]) and
            (not p3 or res1[i] == res3[i]) and
            (not p4 or res1[i] == res4[i])):
            same_responses.append(i)
    
    question_num = np.random.choice(same_responses)
    response = res1[question_num]
    
    return question_num, response

def writeMatchesToFile(matches, people, out_path, first=True, last=True):
    """
    params:
        first - True if this is the first set of matches written to the file
        last - True if this is the last set of matches written to the file
    """
    mode = 'w' if first else 'a'
    with open(out_path, mode) as f:
        if first:
            f.write(MATCH_HEADER)
        
        community_score = 0.0
        
        for match in matches:
            p1_id, p2_id = match[0]
            
            p1 = people[p1_id]
            p2 = people[p2_id]
            
            p1_info = p1['meta data']
            p2_info = p2['meta data']
            
            question_num, response = findSimilarResponse(p1, p2)
            
            print("Person " + f"{p1_id}".rjust(3) + f" matched with " + \
                  "Person " + f"{p2_id}".rjust(3) + \
                  f" (score: {match[1]:.4%})")
            f.write(f"{p1_id},{p1_info['name']},{p1_info['email']}," + \
                    f"{p1_info['sport name']},{p1_info['year']}," + \
                    f"{p1['athlete mingle']['same grade']}," + \
                    f"{p2_id},{p2_info['name']},{p2_info['email']}," + \
                    f"{p2_info['sport name']},{p2_info['year']}," + \
                    f"{p2['athlete mingle']['same grade']}," + \
                    f"{match[1]:.4%},{question_num},{response}\n")
            
            community_score += match[1]
            
        community_score /= len(matches)
        
        message = f"\nCommunity score: {community_score:.4%}\n"
        print(message)
        f.write(message)

        if not last:
            f.write('\n')

################################## Matching ###################################

def makePairMatches(people):
    """
    Returns an array of matches where each match is a list of length 2. The
    frist element of the list is a pair of indices into the PEOPLE array. The
    second element of the list is the computed score for that pair.
    
    This algorithm uses a greedy algorithm: find the highest match, then pair
    those two and remove them from set of matches.
    """
    original_scores = computeSimilarities(people)
    scores = copy.deepcopy(original_scores)
    
    print('Creating matches')
    
    unmatched = set(range(len(people)))
    matches = []
    while (len(unmatched) > 1):
        ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
        matches.append([ind, scores[ind]])
        
        for i in ind:
            assert i in unmatched, 'Ran out of matches. Please re-run.'
            
            unmatched.remove(i)
            scores[i, :] = -1
            scores[:, i] = -1
            
    if len(unmatched) == 1:
        print('WARNING: Someone is left unmatched.')
        print(json.dumps(people[list(unmatched)[0]], indent=4))
    
    return matches

def matchGroups(group):
    """
    Create pair matches. We must respect match requests for people who put a
    friend's name down. Thus, there is some additional processing to figure out
    hard matches. Then, the rest of the people are matched into pairs using the
    same algorithm for one-on-ones.
    
    Afterwards, we will go through the matches by hand, cleaning any poor
    matches. Lastly, we will manually combine pairs together to create groups
    of four.
    """
    print(f"\n{'=' * 10} Matching groups {'=' * 10}")
    
    """ Aggregate all friend requests """
    # dictionary of name -> list of friends' names
    friend_dict = {person['meta data']['name']: [] for person in group}
    
    for person in group:
        for friend in person['athlete mingle']['friends']:
            # Someone put down a friend but friend signed up for 1-on-1
            if friend not in friend_dict:
                print(f"{friend} (1-on-1)".ljust(25) + f"nominated by " + \
                      f"{person['meta data']['name']} (group)")
                continue
            # add person and friend to corresponding element in friend_dict
            if friend not in friend_dict[person['meta data']['name']]:
                friend_dict[person['meta data']['name']].append(friend)
            if person['meta data']['name'] not in friend_dict[friend]:
                friend_dict[friend].append(person['meta data']['name'])
    
    """ Separate into people who put friend requests and people who didn't """
    friend_requests = []
    no_friend_requests = []
            
    for name, friends in friend_dict.items():
        if len(friends) == 0:
            no_friend_requests.append(findPerson(name, group))
        else:
            # if the person is already included, don't create a duplicate match
            for person in friend_requests:
                if person['meta data']['name'] == friends[0]:
                    break
            else:
                """
                Only get the first friend. Add the person and friend to the
                list. Append the indices to matches with the computed score.
                """
                p1, p2 = findPerson(name, group), findPerson(friends[0], group)
                friend_requests.extend([p1, p2])

    # matches with no friend requests
    matches = makePairMatches(no_friend_requests)
    writeMatchesToFile(matches, no_friend_requests,
                       f'{OUT_PATH}matches_group.csv', last=False)
    
    # matches with friend requests
    matches = []
    
    for i in range(0, len(friend_requests) - 1, 2):
        p1, p2 = friend_requests[i:i+2]
        score = computeSimilarity(p1, p2, same_kind_ok=True,
                                  same_sport_ok=True)
        matches.append([(i, i + 1), score])
        
    writeMatchesToFile(matches, friend_requests,
                       f'{OUT_PATH}matches_group.csv', first=False)

def matchOneOnOnes(one_on_one):
    print(f"\n{'=' * 10} Matching one-on-ones {'=' * 10}")
    
    matches = makePairMatches(one_on_one)
    writeMatchesToFile(matches, one_on_one, f'{OUT_PATH}matches_1-on-1.csv')
    
    # Print kind distributions
    ok_count = 0
    male_male_count = 0
    female_female_count = 0
    coed_count = 0
        
    for match in matches:
        p1_id, p2_id = match[0]
        
        p1 = one_on_one[p1_id]
        p2 = one_on_one[p2_id]
        
        kind1 = SPORT_DATA[p1['meta data']['sport name']]['kind']
        kind2 = SPORT_DATA[p2['meta data']['sport name']]['kind']
        if kind1 != kind2:
            ok_count += 1
        elif kind1 == "Men":
            male_male_count += 1
        elif kind2 == "Women":
            female_female_count += 1
        elif kind2 == "Co-Ed":
            coed_count += 1
            
    print(f'{ok_count} male-female')
    print(f'{male_male_count} male-male')
    print(f'{female_female_count} female-female')
    print(f'{coed_count} coed-coed')

def main():
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(PLOT_PATH, exist_ok=True)
    
    people = loadData()
    analyzeData(people)
    one_on_one, group = separatePairsAndGroups(people)
    matchOneOnOnes(one_on_one)
    matchGroups(group)

if __name__ == '__main__':
    main()