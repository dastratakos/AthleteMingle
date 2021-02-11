"""
file: match.py
date: 2/10/21
author: Dean Stratakos
----------------------

First 17 columns of each row:

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

############### Meta-Data ###############
17: First, let's get a few things out of the way
What grade are you in?
18: What sport do you do?
19: Major? (or interest if no major yet)
20: Do you want to be placed in a group with one teammate or friend? You should both put each other's name for this question!
21: Do you want to be placed with people in your grade?
22: Would you rather do a 1-on-1?
23: Would you be interested in a student-athlete speed-dating event in the future?
"""

import csv
import json

import numpy as np

def loadData(filename):
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
    print(json.dumps(people[-1], indent=4))
    
    return people

def main():
    people = loadData('data/Athlete Mingle_February 10, 2021_21.20.csv')

if __name__ == '__main__':
    main()