#! /usr/bin/env python2.7


import csv
import flickr
import urllib, urlparse
import os
import sys

def search(loc):
    # downloading image data
    for page in range(1, 9):
        f = flickr.photos_search(text=name+' clouds', per_page=500, page=page)
        urllist = [] #store a list of what was downloaded

        # downloading images
        for k in f:
            try:
                url = k.getURL(size='Medium', urlType='source')
                urllist.append(url)
                image = urllib.URLopener()
                if not os.path.exists(loc+os.path.basename(urlparse.urlparse(url).path)):
                    image.retrieve(url, loc+os.path.basename(urlparse.urlparse(url).path))
                    print 'downloading:', url
                else:
                    print 'already downloaded ' + image + ' skipping'
            except:
                print 'error on ' + url + ' skipping'
                continue

        # write the list of urls to file       
        fl = open('urllist.txt', 'w')
        for url in urllist:
            fl.write(url+'\n')
        fl.close()


with open('../cloud-types.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        name = row[0]
        loc = './'+name+'/'
        if not os.path.exists(loc):
            os.mkdir(loc)

        search(loc)
