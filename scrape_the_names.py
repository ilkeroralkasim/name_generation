import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import random
import time

name_list = []

urls = ['https://www.behindthename.com/names']
urls = urls + [urls[0] + '/' + str(i) for i in range(2, 78)]
filenames = ['page' + str(i) + ".txt" for i in range(77)]

f_u = zip(urls, filenames)

for url, filename in f_u:
    with open(filename, "w") as target_file:
        print('Writing file: ', filename)
        web_page = requests.get(url)
        time.sleep(5)
        soup = BeautifulSoup(web_page.text, 'html.parser')
        results = soup.find_all('span', attrs={'class':'listname'})
        for result in results:
            n = result.find('a').text
            name_list.append(n)
            target_file.write(n + '\n')


print(len(name_list))
print(name_list[20:30])
uni_list = sorted(set(name_list))

print(len(uni_list))
print(uni_list)
