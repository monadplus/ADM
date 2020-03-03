from data import Data
from naivebayes import NaiveBayes
from maxapost import MaxAPost
from confmat import ConfMat

# filename = "./datasets/weatherNominalTr.txt"
filename = "datasets/titanicTr.txt"
# filename = "datasets/cmcTr.txt"

d = Data(filename, 75) # 75% for training
d.report()

pr = NaiveBayes(d)
# pr = MaxAPost(d)
pr.train()

cm = ConfMat(pr.clsscnts)
for (v,c_true) in d.test_set:
        c_pred = pr.predict(v)[0]
        # print(v, c_pred, "( true class:", c_true, ")")
        cm.mat[c_pred,c_true] += 1

# pr.show()
cm.report()


## Not all samples are classifiable in MAP because you need to have for each class Y the attribute X = (X1,X2,X3..)
## For example on dataset cmcTr.txt, MAP will fail.

# print(pr.predict(("Class:Crew","Sex:Female","Age:Child")))
# print(pr.predict(("Class:1st","Sex:Female","Age:Child")))
