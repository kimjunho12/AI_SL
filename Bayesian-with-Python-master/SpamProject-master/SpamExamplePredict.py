import pickle
import os
from konlpy.tag import Okt

os.chdir(os.path.dirname(__file__))

with open('./model/spam_mail_filter.model', 'rb') as file:
    spamFilterModel = pickle.load(file)

print("스팸메일 아닌 예시")
hamExam = open("./example/hamExam.txt", 'r').read()
print(hamExam)
pre = spamFilterModel.predict(hamExam)
print("결과 =", pre)

print("-----------------------------------")

print("스팸메일인 예시")
spamExam = open("./example/spamExam.txt", 'r').read()
print(spamExam)
pre = spamFilterModel.predict(spamExam)
print("결과 =", pre)
