import csv

def proprocess(input_file, output_file):
    f = open(input_file)
    a = {}
    b = {}
    for i in f.readlines():
        s = i[:-1].split("::")
        if s[0] not in a:
            a[s[0]] = []
        a[s[0]].append((s[1], s[3]))
        if s[1] not in b:
            b[s[1]] = []
        b[s[1]].append(s[0])

    users = {}
    items = {}
    with open(output_file, mode='w', newline='') as w_file:
        csv_writer = csv.writer(w_file)#, delimiter=' ')

        for i in a.keys():
            if len(a[i]) >=5 :
                for k,t in sorted(a[i], key=lambda x: x[1]):
                    if len(b[k]) >= 5:
                        if k not in items:
                            items[k] = len(items)+1
                        if i not in users:
                            users[i] = len(users)+1
                        csv_writer.writerow([users[i], items[k], t])
                        #csv_writer.writerow([users[i], items[k]])


if __name__ == "__main__":
    f = 'data/raw/movieln_1m/ratings.dat'
    output_file = 'data/ml-1m_timestamps.txt'
    proprocess(f, output_file)
