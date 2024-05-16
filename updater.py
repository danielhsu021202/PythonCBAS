import requests
import os

# Test downloading a file from github

dest = os.path.join(os.getcwd(), 'downloadtest')
os.makedirs(dest, exist_ok=True)

url = 'https://raw.github.com/Ayushparikh-code/Web-dev-mini-projects/main/Age%20Calculator/result.png'
r = requests.get(url, allow_redirects=True)

with open(os.path.join(dest, 'output.png'), 'wb') as f:
    f.write(r.content)


