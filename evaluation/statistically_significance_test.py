import sys
import pandas as pd
from scipy import stats

def main(path, file1, file2):
    pval = 0.01
    data1 = pd.read_csv(path+file1, sep=';')
    data2 = pd.read_csv(path+file2, sep=';')

    ttest(data1, data2, 'HR@1', pval)
    ttest(data1, data2, 'NDCG@5', pval)
    ttest(data1, data2, 'HR@5', pval)
    ttest(data1, data2, 'NDCG@10', pval)
    ttest(data1, data2, 'HR@10', pval)
    ttest(data1, data2, 'MRR', pval)


def ttest(data1, data2, metric, pval):
    result1 = data1[metric]
    result2 = data2[metric]
    statistic, pvalue = stats.ttest_rel(result1, result2)
    print("statistic= "+str(statistic)+", pvalue= "+str(pvalue))
    if pvalue < pval:
        print(metric+": statistically significant! (reject the null hypothesis)")
    else:
        print(metric+": cannot reject the null hypothesis (which is the mean scores are equal)")

if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('File or folder expected.')