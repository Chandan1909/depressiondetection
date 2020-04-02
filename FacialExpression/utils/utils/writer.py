import csv
fh=open('emotions.csv','a',newline='')
writer=csv.writer(fh)
for i in b:
    writer.writerow(i)
fh.close()