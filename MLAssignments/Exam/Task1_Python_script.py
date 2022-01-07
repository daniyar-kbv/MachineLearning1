# Python script to scrape an article given the url of the article and store the extracted text in a file
# Url: https://medium.com/@subashgandyer/papa-what-is-a-neural-network-c5e5cc427c7

from bs4 import BeautifulSoup
import os
import requests
import re
import sys


# function to get the html source text of the medium article
def get_page():
	global url

	url = input('Enter url of a medium article: ')

	# handling possible error
	if not re.match(r'https?://medium.com/', url):
		print('Please enter a valid website, or make sure it is a medium article')
		sys.exit(1)

	res = requests.get(url)

	res.raise_for_status()
	soup = BeautifulSoup(res.text, 'html.parser')
	return soup


# function to remove all the html tags and replace some with specific strings
def clean(text):
	rep = {"<br>": "\n", "<br/>": "\n", "<li>":  "\n"}
	rep = dict((re.escape(k), v) for k, v in rep.items())
	pattern = re.compile("|".join(rep.keys()))
	text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
	text = re.sub('<(.*?)>', '', text)
	return text


def collect_text(soup):
	text = f'url: {url}\n\n'
	para_text = soup.find_all('p')
	print(f"paragraphs text = \n {para_text}")
	for para in para_text:
		text += f"{para.text}\n\n"
	return text


# function to save file in the current directory
def save_file(text):
	directory_path = './scraped_articles'

	if not os.path.exists(directory_path):
		os.mkdir(directory_path)

	article_name = url.split("/")[-1]
	file_path = f'{directory_path}/{article_name}.txt'

	with open(file_path, 'w') as file:
		file.write(text)

	print(f'File saved in directory {file_path}')


if __name__ == '__main__':
	text = collect_text(get_page())
	save_file(text)
	# Instructions to Run this python code
	# Give url as https://medium.com/@subashgandyer/papa-what-is-a-neural-network-c5e5cc427c7