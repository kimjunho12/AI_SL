import pickle
from konlpy.tag import Okt
import os

os.chdir(os.path.dirname(__file__))

with open('./model/spam_mail_filter.model', 'rb') as file:
    spamFilterModel = pickle.load(file)


print("검사할 내용을 입력해주세요")
contents = []
while True:
    content = input()
    if content:
        contents.append(content)
    else:
        break
text = '\n'.join(contents)
pre = spamFilterModel.predict(text)
print("결과 =", pre)
print()
