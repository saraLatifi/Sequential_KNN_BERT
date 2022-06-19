import csv
import sys

def process_input(csv_file_path):

    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        best_results={}
        for i, row in enumerate(csv_reader):
            if i == 0:
                for j,metric in enumerate(row):
                    if j >0:
                        best_results[j] =[ [0, 0, metric]]
            else:
                 for j,metric in enumerate(row):
                     if j >0  and j< 8 and float(metric) > best_results[j][0][1]:
                        best_results[j] = [[row[0], float(metric), best_results[j][0][2]]]
                     elif j >0  and j< 8 and float(metric) == best_results[j][0][1]:
                        best_results[j].append([[row[0], float(metric), best_results[j][0][2]]])
        for k,j in best_results.items():
            for x in j:
                print (",".join([str(i) for i in x]))
            

if __name__ == "__main__":
    # execute only if run as a script
    process_input(sys.argv[1])
