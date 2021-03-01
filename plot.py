"""
file: plot.py
date: 2/10/21
author: Dean Stratakos
----------------------
Helper functions for visualizing data from Athlete Mingle.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sb

SPORT_DATA = {}
with open('sports.json') as f:
    SPORT_DATA = json.load(f)

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
kind_labels = ['Men', 'Women', 'Co-Ed']

def autolabel(ax, rects):
    """
    Helper function.
    Attach a text label above each bar in *rects*, displaying its height.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),              # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')

def autolabelh(ax, rects):
    """
    Helper function.
    Attach a text label above each bar in *rects*, displaying its width.
    """
    for rect in rects:
        width = rect.get_width()
        ax.annotate('{}'.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),              # 3 points horizontal offset
                    textcoords='offset points',
                    ha='left', va='center')

def plotHeatMap(scores, out_path):
    print('Plotting heat map')
    
    plt.subplots(figsize=(11, 9))
    sb.heatmap(scores, annot=True)
    plt.savefig(out_path)
    plt.close()
    
def plotYears(years, out_path, interactive=False):
    print('Plotting years')
    
    counts = [years.count(y) for y in year_labels]
    
    plt.bar(range(len(counts)), counts, color='#Bf0A30')
    plt.xlabel('Year')
    plt.ylabel('Responses')
    plt.title('Athlete Mingle Year Distribution')

    plt.xticks(range(len(counts)), year_labels)

    if interactive: plt.show()
    else: plt.savefig(out_path)
    plt.close()
    
def plotMajors(majors, out_path, interactive=False):
    print('Plotting majors')
    
    counts = [majors.count(label) for label in major_labels]
    
    fig, ax = plt.subplots()
    
    fig.set_size_inches(18.5, 10.5)
    plt.subplots_adjust(left=0.2)
    
    rects = ax.barh(range(len(counts)), counts, color='#Bf0A30')
    ax.invert_yaxis()
    plt.ylabel('Majors')
    plt.xlabel('Responses')
    plt.title('Athlete Mingle Major Distribution')

    plt.yticks(range(len(counts)), major_labels)
    
    autolabelh(ax, rects)

    if interactive: plt.show()
    else: plt.savefig(out_path)
    plt.close()
    
def plotSports(sports, out_path, interactive=False):
    print('Plotting sports')
    
    counts = [sports.count(label) for label in sport_labels]
    
    fig, ax = plt.subplots()
    
    fig.set_size_inches(18.5, 10.5)
    plt.subplots_adjust(left=0.2)
    
    rects = ax.barh(range(len(counts)), counts, color='#Bf0A30')
    ax.invert_yaxis()
    plt.ylabel('Sport')
    plt.xlabel('Responses')
    plt.title('Athlete Mingle Sport Distribution')

    plt.yticks(range(len(counts)), sport_labels)
    
    autolabelh(ax, rects)

    if interactive: plt.show()
    else: plt.savefig(out_path)
    plt.close()
    
def plotKinds(sports, out_path, interactive=False):
    print('Plotting kinds')
    
    counts = [0, 0, 0]
    for res in sports:
        for sport, sport_details in SPORT_DATA.items():
            if sport == res:
                counts[kind_labels.index(sport_details['kind'])] += 1
    
    # fig, ax = plt.subplots()
    
    plt.bar(range(len(counts)), counts, color='#Bf0A30')
    plt.xlabel('Kind')
    plt.ylabel('Responses')
    plt.title('Athlete Mingle Kind Distribution')

    plt.xticks(range(len(counts)), kind_labels)

    if interactive: plt.show()
    else: plt.savefig(out_path)
    plt.close()
    
def plotYearKind(years, sports, out_path, interactive=False):
    print('Plotting years and kinds')
    
    counts = np.zeros((3, len(year_labels)), dtype='intc')
    
    for y, s in zip(years, sports):
        kind = SPORT_DATA[s]['kind']
        counts[kind_labels.index(kind)][year_labels.index(y)] += 1
    
    x = np.arange(len(year_labels))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, counts[0], width, label='Men', color='#Bf0A30')
    rects2 = ax.bar(x, counts[1], width, label='Women', color='#4298B5')
    rects3 = ax.bar(x + width, counts[2], width, label='Co-Ed', color='#7F7776')

    ax.set_ylabel('Responses')
    ax.set_title('Athlete Mingle Year and Kind distributions')
    ax.set_xticks(x)
    ax.set_xticklabels(year_labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    
    fig.tight_layout()

    if interactive: plt.show()
    else: plt.savefig(out_path)
    plt.close()
    
def plotResponses(responses, out_path, interactive=False):
    print('Plotting responses')
    
    fig, axs = plt.subplots(6, 5)
    fig.set_size_inches(30, 11)
    plt.subplots_adjust(left=0.05, right=0.95, hspace=0.6)
    
    for q in range(len(responses[0])):
        freq, bins, patches = axs[q // 5, q % 5].hist(responses[:, q], bins=5)
        axs[q // 5, q % 5].set_title(f'Question {q}')
        axs[q // 5, q % 5].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[q // 5, q % 5].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[q // 5, q % 5].set_ylim((0, 300))
        
        bin_centers = np.diff(bins) * 0.5 + bins[:-1]

        n = 0
        for fr, x, patch in zip(freq, bin_centers, patches):
            height = int(fr)
            axs[q // 5, q % 5].annotate("{}".format(height),
                xy = (x, height),             # top left corner of histogram bar
                xytext = (0, 10),             # offset label position above bar
                textcoords = "offset points", # offset (in points) from *xy* value
                ha = 'center', va = 'bottom',
                fontsize=10)
            axs[q // 5, q % 5].annotate(f"{height / 327:.1%}",
                xy = (x, height),             # top left corner of histogram bar
                xytext = (0, 0.2),            # offset label position above bar
                textcoords = "offset points", # offset (in points) from *xy* value
                ha = 'center', va = 'bottom',
                fontsize=8)
            n = n + 1
    
    if interactive: plt.show()
    else: plt.savefig(out_path)
    plt.close()