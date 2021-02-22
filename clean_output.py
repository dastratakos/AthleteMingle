import csv

filename = '1-on-1 Matches FINAL.csv'
new_filename = '1-on-1 Matches FINAL CLEANED.csv'

with open(filename) as f:
    header, *responses = csv.reader(f)
    
with open(new_filename, 'w') as f:
    f.write(','.join(header[1:3]) + ',' + header[5] + ',' + ','.join(header[3:5]) + ',')
    f.write(','.join(header[7:9]) + ',' + header[11] + ',' + ','.join(header[9:11]) + ',')
    f.write(','.join(header[-3:]) + '\n')
    for row in responses[:-2]:
        old_score = float(row[-3].strip('%')) / 100
        old_score = .5 if old_score < 0 else old_score
        new_score = .5115 * old_score + .5561
        new_score = f'{new_score:.4%}'
        f.write(row[1].title() + ',' + row[2].lower() + ',' + row[5] + ',' + ','.join(row[3:5]) + ',')
        f.write(row[7].title() + ',' + row[8].lower() + ',' + row[11] + ',' + ','.join(row[9:11]) + ',')
        f.write(new_score + ',' + ','.join(row[-2:]) + '\n')