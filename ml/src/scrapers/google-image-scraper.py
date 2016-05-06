#!/usr/bin/env python3.4

import sys
import json
import time
import os
import requests
import ipgetter
from PIL import Image
from io import BytesIO

def main():
    # Get the list of arguments
    arg_list = sys.argv

    # Ensure sufficient arguments
    if len(arg_list) < 3:
        print ('Usage: ./google_image_scraper <save_path> <keywords>...')
        sys.exit()

    # Get the location to store the images
    path = arg_list[1]

    # Join keywords with %20 to comply with Google API requirements
    keywords = '%20'.join(arg_list[2:])

    # The base url and path. Appends public IP as per Google API request
    URL = 'https://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=' \
        + keywords + '&start=%d&userip=' + ipgetter.myip()
    PATH = os.path.join(path, arg_list[2] + '-dataset')

    # Check if directory already exists
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    count = 0
    while count < 60:
        request = requests.get(URL % count)

        for image_info in json.loads(request.text)['responseData']['results']:
            url = image_info['unescapedUrl']

            try:
                image = requests.get(url)
            except requests.exceptions.ConnectionError:
                print ('Could not download %s' % url)
                continue

            title = image_info['titleNoFormatting'].replace('/', '').replace('\\', '')

            file = open(os.path.join(PATH, '%s.jpg') % title, 'w')
            try:
                Image.open(BytesIO(image.content)).save(file, 'JPEG')
            except IOError:
                print ('Could not save %s' % url)
                continue
            finally:
                file.close()

        print (count)
        count = count + 4

        # Sleep to prevent IP blocking from Google
        time.sleep(1.5)


if __name__ == '__main__':
    main()
