import sys
import Textrank_korea_patent_highlighter
import os

print("\n\n\n\n\n")


file_paths = sys.argv[1:]  # the first argument is the script itself

print('변환목록>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}개\n'.format(len(file_paths)))

for p in file_paths:
    base = os.path.basename(p)
    print(base)

#
# user = input('변환하시겠습니까?(y/n)')

print('변환start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')

for p in file_paths:
    Textrank_korea_patent_highlighter.doc_summury2(p)