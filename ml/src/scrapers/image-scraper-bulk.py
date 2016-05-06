#!/usr/bin/env python3.4

import sys
import os
import threading
import requests
import urllib
import time
import csv
from GoogleScraper import scrape_with_config
from GoogleScraper import GoogleSearchError
from GoogleScraper.database import ScraperSearch
from GoogleScraper.database import SERP
from GoogleScraper.database import Link


PARAMETERS = ['<path>', '<keyword>', '<output_filename>']
USAGE = 'Usage: ./image_scraper ' + ' '.join(PARAMETERS)
BASE_DATA_PATH = '/home/javier/Developer/projects/cloudify/data/new-data/'


class ConfigFactory():
    """ Creates the configuration file """
    @staticmethod
    def create_config(keyword):
        config = {
            'SCRAPING': {
                'keyword': keyword,
                'search_engines': 'yandex,google,bing,yahoo',
                'search_type': 'image',
                'scrapemethod': 'selenium',
                'num_pages_for_keyword': '5',
            }
        }
        return config
        

class FetchResource(threading.Thread):
    """ Grabs a web resource and stores it in the target directory"""
    def __init__(self, target, urls):
        super().__init__()
        self.target = target
        self.urls = urls


    def run(self):
        for url in self.urls:
            url = urllib.parse.unquote(url)
            with open(os.path.join(self.target, url.split('/')[-1]), 'wb') as file:
                try:
                    content = requests.get(url, timeout=10).content
                    file.write(content)
                except Exception:
                    pass
                print('[+] Fetched {}'.format(url))


def main(arg_list):
    # Get the arguments
    path = arg_list[0]
    keyword = arg_list[1]

    # Create our target directory if it doesn't exist
    if not os.path.exists(path):
        os.mkdir(path)

    # Create our configuration file
    config = ConfigFactory.create_config(keyword)

    try:
        sqlalchemy_session = scrape_with_config(config)
    except GoogleSearchError:
        print('Error!')

    image_urls = []
    search = sqlalchemy_session.query(ScraperSearch).all()[-1]

    for serp in search.serps:
        image_urls.extend([link.link for link in serp.links])

    print('[i] Going to scrape {num} images and saving them in "{dir}"'.format(
        num=len(image_urls),
        dir=path
    ))

    thread_count = 100
    threads = [FetchResource(path, []) for i in range(thread_count)]

    while image_urls:
        for thread in threads:
            try:
                thread.urls.append(image_urls.pop())
            except IndexError:
                break

    threads = [thread for thread in threads if thread.urls]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    with open('../../data/new-data/cloud-types.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            name = row[0]

            if not os.path.exists(BASE_DATA_PATH+name):
                print('Scraping images for: ' + name)

                main([BASE_DATA_PATH+name, name])
                main([BASE_DATA_PATH+name, name+' flickr'])
                main([BASE_DATA_PATH+name, name+' pixabay'])
                main([BASE_DATA_PATH+name, name+' clouds'])
                main([BASE_DATA_PATH+name, name+' clouds flickr'])
                main([BASE_DATA_PATH+name, name+' clouds pixabay'])
