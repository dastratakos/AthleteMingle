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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sb

SPORTS = {}
with open('sports.json') as f:
    SPORTS = json.load(f)

print(SPORTS)

def loadData(filename):
    print(f"{'=' * 10} Loading data {'=' * 10}")
    
    people = []
    
    with open(filename) as f:
        question_ids, questions, _, *responses = csv.reader(f)
        
        # for i, column in enumerate(questions):
        #     print(f"{i}: {column}")
        
        for person_id, row in enumerate(responses):
            person = {
                'id': person_id,
                'start date': row[0],
                'end date': row[1],
                'qualtrics_id': row[8],
                'meta data': {
                    'name': row[17],
                    'email': row[18],
                    'year': int(row[19]),
                    # 'gender': sports[row[20]]["gender"],
                    # 'sport': sports[row[20]]["sport"],
                    'sport': int(row[20]),
                    'major': int(row[21]),
                },
                'athlete mingle': {
                    '1-on-1': int(row[22]),
                    'friend': row[23],
                    'same grade': int(row[24]),
                },
                'responses': [int(x) for x in row[25:54]],
                'speed-dating': int(row[54])
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
    
    ########## Analyze years ##########
    
    years = [person['meta data']['year'] for person in people]
    labels = ['Frosh', 'Sophomore', 'Junior', 'Senior', '5th Year/Coterm']
    counts = [years.count(i) for i in range(1, len(labels) + 1)]
    
    plt.bar(range(len(counts)), counts, color='#Bf0A30')
    plt.xlabel('Year')
    plt.ylabel('Responses')
    plt.title('Athlete Mingle Year Distribution')

    plt.xticks(range(len(counts)), labels)

    plt.savefig('output/years.png')
    
    ########## Get sports ##########
    
    res_sports = []
    
    with open('data/Athlete Mingle_February 13, 2021_22.04.csv') as f:
        question_ids, questions, _, *responses = csv.reader(f)
        
        res_sports = [row[20] for row in responses]
        
    ########## Analyze sports ##########
    
    labels = [sport['sport'] for sport in SPORTS.values()]
    counts = [res_sports.count(label) for label in labels]
    
    fig, ax = plt.subplots()
    
    # TODO: set plt dimensions
    
    ax.barh(range(len(counts)), counts, color='#Bf0A30')
    ax.invert_yaxis()
    plt.ylabel('Sport')
    plt.xlabel('Responses')
    plt.title('Athlete Mingle Sport Distribution')

    plt.yticks(range(len(counts)), labels)

    # plt.savefig('output/sports.png')
    plt.show()
    
    ########## Analyze sports ##########
    
    labels = ['M', 'W', 'B']
    counts = [0, 0, 0]
    for res in res_sports:
        for sport in SPORTS.values():
            if sport['sport'] == res:
                counts[labels.index(sport['gender'])] += 1
    
    fig, ax = plt.subplots()
    
    plt.bar(range(len(counts)), counts, color='#Bf0A30')
    plt.xlabel('Gender')
    plt.ylabel('Responses')
    plt.title('Athlete Mingle Gender Distribution')

    plt.xticks(range(len(counts)), ['Men', 'Women', 'Co-Ed'])

    # plt.savefig('output/sports.png')
    # plt.show()
    
    ########## Analyze year/gender ##########
    
    labels = ['Frosh', 'Sophomore', 'Junior', 'Senior', '5th Year/Coterm']
    men_count = [0] * len(labels)
    women_count = [0] * len(labels)
    both_count = [0] * len(labels)
    
    for y, s in zip(years, res_sports):
        gender = None
        for sport in SPORTS.values():
            if sport['sport'] == s:
                gender = sport['gender']
                break
                
        if gender == 'M':
            men_count[y - 1] += 1
        elif gender == 'W':
            women_count[y - 1] += 1
        if gender == 'B':
            both_count[y - 1] += 1

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, men_count, width, label='Men', color='#Bf0A30')
    rects2 = ax.bar(x, women_count, width, label='Women', color='#4298B5')
    rects3 = ax.bar(x + width, both_count, width, label='Co-Ed', color='#7F7776')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Responses')
    ax.set_title('Athlete Mingle Year and Gender distributions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()

    plt.show()
    
    ########## Analyze response distribution ##########
    
    responses = np.array([person['responses'] for person in people])
    print(responses)
    
    fig, axs = plt.subplots(6, 5)
    fig.set_size_inches(18.5, 10.5)
    plt.subplots_adjust(left=0.05, right=0.95, hspace=0.6)
    
    for q in range(len(responses[0])):
        # TODO: add percentage labels for each bin (plt.annotate)
        axs[q // 5, q % 5].hist(responses[:, q], bins=5)
        axs[q // 5, q % 5].set_title(f'Question {q}')
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
    filenames = [f for f in os.listdir('data') if f.startswith('Athlete')]
    filename = 'data/' + max(filenames)
    people = loadData(filename)
    people = cleanData(people)
    analyzeData(people)
    # matches = match(people)

if __name__ == '__main__':
    main()