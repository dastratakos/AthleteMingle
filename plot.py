import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sb

def autolabel(ax, rects):
    """
    Helper function.
    Attach a text label above each bar in *rects*, displaying its height.
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')

def autolabelh(ax, rects):
    """
    Helper function.
    Attach a text label above each bar in *rects*, displaying its height.
    """
    for rect in rects:
        width = rect.get_width()
        ax.annotate('{}'.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords='offset points',
                    ha='left', va='center')

def plotHeatMap(scores, out_path):
    # TODO: check if need plt
    plt.subplots(figsize=(11, 9))
    sb.heatmap(scores, annot=True)
    plt.savefig(out_path)
    
def plotResponses(responses, out_path):
    fig, axs = plt.subplots(6, 5)
    fig.set_size_inches(18.5, 10.5)
    plt.subplots_adjust(left=0.05, right=0.95, hspace=0.6)
    
    for q in range(len(responses[0])):
        # TODO: add percentage labels for each bin (plt.annotate)
        axs[q // 5, q % 5].hist(responses[:, q], bins=5)
        axs[q // 5, q % 5].set_title(f'Question {q}')
        axs[q // 5, q % 5].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[q // 5, q % 5].yaxis.set_major_locator(MaxNLocator(integer=True))
        
    plt.savefig(out_path)
    
def plotYearGender(labels, counts, out_path):
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, counts[0], width, label='Men', color='#Bf0A30')
    rects2 = ax.bar(x, counts[1], width, label='Women', color='#4298B5')
    rects3 = ax.bar(x + width, counts[2], width, label='Co-Ed', color='#7F7776')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Responses')
    ax.set_title('Athlete Mingle Year and Gender distributions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    
    fig.tight_layout()

    # plt.show()
    plt.savefig(out_path)
    
def plotGenders(labels, counts, out_path):
    fig, ax = plt.subplots()
    
    plt.bar(range(len(counts)), counts, color='#Bf0A30')
    plt.xlabel('Gender')
    plt.ylabel('Responses')
    plt.title('Athlete Mingle Gender Distribution')

    plt.xticks(range(len(counts)), labels)

    # plt.show()
    plt.savefig(out_path)
    
def plotSports(labels, counts, out_path):
    fig, ax = plt.subplots()
    
    # TODO: set plt dimensions
    
    rects = ax.barh(range(len(counts)), counts, color='#Bf0A30')
    ax.invert_yaxis()
    plt.ylabel('Sport')
    plt.xlabel('Responses')
    plt.title('Athlete Mingle Sport Distribution')

    plt.yticks(range(len(counts)), labels)
    
    autolabelh(ax, rects)

    # plt.show()
    plt.savefig(out_path)
    
def plotYears(labels, counts, out_path):
    plt.bar(range(len(counts)), counts, color='#Bf0A30')
    plt.xlabel('Year')
    plt.ylabel('Responses')
    plt.title('Athlete Mingle Year Distribution')

    plt.xticks(range(len(counts)), labels)

    plt.savefig(out_path)