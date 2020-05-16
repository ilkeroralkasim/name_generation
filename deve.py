import re

s_file_names = ['page' + str(i) + '.txt' for i in range(77)]
t_file_names = ['cleaned/page_clean' + str(i) + '.txt' for i in range(77)]
s_t_pair = zip(s_file_names, t_file_names)

re_n = re.compile('[1-9()\']')

cleaned_names = []

for s in s_file_names:
    with open(s, 'r') as source:
        for line in source:
                cleaned_names.append(re.sub(re_n, '', line))

print(len(cleaned_names))
cleaned_names = sorted(set(cleaned_names))

with open('names_final.txt', 'w') as f_target:
    for n in cleaned_names:
        f_target.write(n)