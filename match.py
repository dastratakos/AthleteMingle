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
import json
import random

import numpy as np
from tqdm import tqdm

from plot import *

TEXT_FILE = 'data/Athlete Mingle_February 18, 2021_11.25.csv'
NUMBER_FILE = 'data/Athlete Mingle_February 18, 2021_11.24.csv'
OUT_PATH = 'output/02-20/'

SPORT_DATA = {}
with open('sports.json') as f:
    SPORT_DATA = json.load(f)

def loadData():
    print(f"{'=' * 10} Loading data {'=' * 10}")
        
    people = []
    skip_ids = []
    
    with open(NUMBER_FILE) as f:
        _, column_headers, _, *responses = csv.reader(f)
        
        # for i, column in enumerate(column_headers[25:54]):
        #     print(f"{i}: {column}")
        
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

    # random.seed(42)
    
    # Randomize people
    random.shuffle(people)
            
    print(f'There were {len(people)} responses ' + \
          f'and {len(skip_ids)} unfinished surveys.')
    # print(json.dumps(people[-1], indent=4))
    
    return people

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
    sport_labels = SPORT_DATA.keys()
    gender_labels = ['Men', 'Women', 'Co-Ed']
    
    
    ########## Analyze years ##########
    
    years = [person['meta data']['year'] for person in people]
    counts = [years.count(y) for y in year_labels]
    
    # plotYears(year_labels, counts, out_path=f'{OUT_PATH}/years.png')
        
    ########## Analyze majors ##########
    
    majors = [person['meta data']['major'] for person in people]
    counts = [majors.count(label) for label in major_labels]
    
    # plotMajors(major_labels, counts, out_path=f'{OUT_PATH}majors.png')
    
    ########## Analyze sports ##########
    
    sports = [person['meta data']['sport name'] for person in people]
    counts = [sports.count(label) for label in sport_labels]
    
    # plotSports(sport_labels, counts, out_path=f'{OUT_PATH}sports.png')
    
    ########## Analyze genders ##########
    
    counts = [0, 0, 0]
    for res in sports:
        for sport, sport_details in SPORT_DATA.items():
            if sport == res:
                counts[gender_labels.index(sport_details['gender'])] += 1
    
    # plotGenders(gender_labels, counts, out_path=f'{OUT_PATH}genders.png')
    
    ########## Analyze year/gender ##########
    
    counts = np.zeros((3, len(year_labels)), dtype='intc')
    
    for y, s in zip(years, sports):
        gender = None
        for sport in SPORT_DATA.values():
            if sport['sport'] == s:
                gender = sport['gender']
                break
            
        counts[gender_labels.index(gender)][year_labels.index(y)] += 1

    # plotYearGender(year_labels, counts, out_path=f'{OUT_PATH}yearsGenders.png')
    
    ########## Analyze response distribution ##########
    
    responses = np.array([person['responses'] for person in people])
    
    plotResponses(responses, out_path=f'{OUT_PATH}responses.png')

def computeSimilarity(p1, p2, same_gender_ok=False, same_sport_ok=False):
    """
    Returns the percentage of questions that were answered the same.
    """
    # check if p1 and p2 are the same person
    if p1['qualtrics id'] == p2['qualtrics id']: return -1.5

    penalties = 0

    if not same_gender_ok:
        gender1 = SPORT_DATA[p1['meta data']['sport name']]['gender']
        gender2 = SPORT_DATA[p2['meta data']['sport name']]['gender']
        if gender1 == gender2 and random.random() < 0.7:
            penalties += -0.4

    if p1['athlete mingle']['same grade'] or p2['athlete mingle']['same grade']:
        if p1['meta data']['year'] != p2['meta data']['year']:
            penalties += -0.5

    if not same_sport_ok:
        # check if they are on the same team
        if p1['meta data']['sport name'] == p2['meta data']['sport name']:
            penalties += -0.3
        
    # check if they are on counterpart teams
    counterpart1 = SPORT_DATA[p1['meta data']['sport name']].get('counterpart', 'none1')
    counterpart2 = SPORT_DATA[p2['meta data']['sport name']].get('counterpart', 'none2')
    if (counterpart1 == p2['meta data']['sport name'] or 
        counterpart2 == p1['meta data']['sport name']):
        penalties += -0.1
    
    if penalties < 0:
        return penalties
    
    responses1 = np.array(p1['responses'])
    responses2 = np.array(p2['responses'])
    
    raw_score = np.count_nonzero(responses1 == responses2) / len(responses1)
    
    # minimize far away grades
    year_labels = ['Frosh', 'Sophomore', 'Junior', 'Senior', '5th year/Coterm']
    if abs(year_labels.index(p1['meta data']['year']) -
           year_labels.index(p2['meta data']['year'])) > 2:
        raw_score *= 0.1
    
    return raw_score

def matchOneOnOnes(one_on_one):
    # compute compatibilities
    print('Computing compatibilities')
    scores = np.zeros((len(one_on_one), len(one_on_one)))
    with tqdm(total=len(one_on_one)) as progress_bar:
        for i, person1 in enumerate(one_on_one):
            progress_bar.update()
            for j, person2 in enumerate(one_on_one):
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
    unmatched = set(range(len(one_on_one)))
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
    
    return matches

def matchGroups(group):
    """
    Make groups of size four.
    """
    # compute compatibilities
    print('Computing compatibilities')
    scores = np.zeros((len(group), len(group)))
    with tqdm(total=len(group)) as progress_bar:
        for i, person1 in enumerate(group):
            progress_bar.update()
            for j, person2 in enumerate(group):
                scores[i][j] = computeSimilarity(person1, person2, same_gender_ok=True)
    
    print(scores)
    # print('Plotting heat map')
    # plotHeatMap(scores, out_path=f'{OUT_PATH}confusion_matrix.png')
    print()
    
    # select best overall groups
    """
    Greedy algorithm: find the highest match, then pair those two and remove
    from set of matches.
    """
    print('Creating matches')
    
    # match requests
    names = [person['meta data']['name'] for person in group]
    
    # dictionary of name -> list of names
    matches = {name: [] for name in names}
    
    # print(json.dumps(matches, indent=4))
    
    for person in group:
        friends = [f.strip()
                   for f in person['athlete mingle']['friend'].split(',')
                   if f != '']
        
        for friend in friends:
            if friend not in matches:
                print(f"{friend} (1-on-1)".ljust(25) + f"nominated by " + \
                      f"{person['meta data']['name']} (group)")
                continue
            if friend not in matches[person['meta data']['name']]:
                matches[person['meta data']['name']].append(friend)
            if person['meta data']['name'] not in matches[friend]:
                matches[friend].append(person['meta data']['name'])
        
    group_match_by_hand = {}
            
    for name, friends in matches.items():
        if len(friends) > 0:
            group_match_by_hand[name] = friends
    
    one_on_one_names = [name for name in matches if name not in group_match_by_hand]
    
    one_on_one = []
    
    for name in one_on_one_names:
        for person in group:
            if person['meta data']['name'] == name:
                one_on_one.append(person)
                break
            
    matches = matchOneOnOnes(one_on_one)
    
    with open(f'{OUT_PATH}matches_group.csv', 'w') as f:
        f.write('p1 id,p1 name,p1 email,p1 sport,p1 year,p1 same grade,' + \
                'p2 id,p2 name,p2 email,p2 sport,p2 year,p2 same grade,' + \
                'score,question,response\n')
        
        community_score = 0.0
        
        for match in matches:
            p1_id, p2_id = match[0]
            
            p1 = one_on_one[p1_id]
            p2 = one_on_one[p2_id]
            
            p1_info = p1['meta data']
            p2_info = p2['meta data']
            
            question_num, response = findSimilarResponse(p1, p2)
            
            print(f"Person {p1_id}".ljust(9) + f" matched with " + \
                  f"Person {p2_id}".ljust(9) + f" ({match[1]:.4%})")
            f.write(f"{p1_id},{p1_info['name']},{p1_info['email']}," + \
                    f"{p1_info['year']},{p1['athlete mingle']['same grade']}," + \
                    f"{p1_info['sport name']}," + \
                    f"{p2_id},{p2_info['name']},{p2_info['email']}," + \
                    f"{p2_info['year']},{p2['athlete mingle']['same grade']}," + \
                    f"{p2_info['sport name']}," + \
                    f"{match[1]:.4%},{question_num},{response}\n")
            
            community_score += match[1]
            
        community_score /= len(matches)
        
        message = f"\nCommunity score: {community_score:.4%}\n"
        print(message)
        f.write(message + '\n')
        
    # remove duplicates
    group_match_with_friends = []
        
    for name, friends in group_match_by_hand.items():
        for p1, p2 in group_match_with_friends:
            if p2['meta data']['name'] == name:
                break
        else:
            friend = friends[0]
            p1, p2 = None, None
        
            for person in group:
                if person['meta data']['name'] == name:
                    p1 = person
                elif person['meta data']['name'] == friend:
                    p2 = person
                if p1 and p2:
                    break
            group_match_with_friends.append((p1, p2))
        
    with open(f'{OUT_PATH}matches_group.csv', 'a') as f:
        community_score = 0.0
        
        for p1, p2 in group_match_with_friends:
            # print(f"p1: {p1['meta data']['name']}".ljust(30) + f"p2: {p2['meta data']['name']}")
            
            p1_id, p2_id = -1, -1
            
            p1_info = p1['meta data']
            p2_info = p2['meta data']
            
            score = computeSimilarity(p1, p2, same_gender_ok=True, same_sport_ok=True)
            question_num, response = findSimilarResponse(p1, p2)
            
            print(f"{p1_info['name']}".ljust(20) + \
                  f" matched with " + f"{p2_info['name']}".ljust(20) + \
                  f" ({score:.4%})")
            f.write(f"{p1_id},{p1_info['name']},{p1_info['email']}," + \
                    f"{p1_info['year']},{p1['athlete mingle']['same grade']}," + \
                    f"{p1_info['sport name']}," + \
                    f"{p2_id},{p2_info['name']},{p2_info['email']}," + \
                    f"{p2_info['year']},{p2['athlete mingle']['same grade']}," + \
                    f"{p2_info['sport name']}," + \
                    f"{score:.4%},{question_num},{response}\n")
            
            community_score += score
            
        community_score /= len(group_match_with_friends)
        
        message = f"\nCommunity score: {community_score:.4%}"
        print(message)
        f.write(message)

def match(people):
    print(f"\n{'=' * 10} Making matches {'=' * 10}")
    
    one_on_one = []
    group = []
    for person in people:
        if person['athlete mingle']['1-on-1']:
            one_on_one.append(person)
        else:
            group.append(person)
    
    matches = matchOneOnOnes(one_on_one)
    
    with open(f'{OUT_PATH}matches_1-on-1.csv', 'w') as f:
        f.write('p1 id,p1 name,p1 email,p1 sport,p1 year,p1 same grade,' + \
                'p2 id,p2 name,p2 email,p2 sport,p2 year,p2 same grade,' + \
                'score,question,response\n')
        
        community_score = 0.0
        
        for match in matches:
            p1_id, p2_id = match[0]
            
            p1 = one_on_one[p1_id]
            p2 = one_on_one[p2_id]
            
            p1_info = p1['meta data']
            p2_info = p2['meta data']
            
            question_num, response = findSimilarResponse(p1, p2)
            
            print(f"Person {p1_id}".ljust(9) + f" matched with " + \
                  f"Person {p2_id}".ljust(9) + f" ({match[1]:.4%})")
            f.write(f"{p1_id},{p1_info['name']},{p1_info['email']}," + \
                    f"{p1_info['year']},{p1['athlete mingle']['same grade']}," + \
                    f"{p1_info['sport name']}," + \
                    f"{p2_id},{p2_info['name']},{p2_info['email']}," + \
                    f"{p2_info['year']},{p2['athlete mingle']['same grade']}," + \
                    f"{p2_info['sport name']}," + \
                    f"{match[1]:.4%},{question_num},{response}\n")
            
            community_score += match[1]
            
        community_score /= len(matches)
        
        message = f"\nCommunity score: {community_score:.4%}"
        print(message)
        f.write(message)
        
    matchGroups(group)

def findSimilarResponse(p1, p2):
    """
    Returns a random similar response.
    """
    responses1 = np.array(p1['responses'])
    responses2 = np.array(p2['responses'])
    
    same_responses = np.argwhere(responses1 == responses2).flatten()
    
    question_num = np.random.choice(same_responses)
    response = responses1[question_num]
    
    return question_num, response

def main():
    people = loadData()
    # analyzeData(people)
    matches = match(people)

if __name__ == '__main__':
    main()


# TODO: squish closer to 100% match
# TODO: TS no FH, Gym

# download question data and percentages and send to data (4) and graphics (1) team