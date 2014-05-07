#!python
import urllib
import urllib2
import json

data = json.dumps({"type" : "lesk", "syn1" : "car#n#1", "syn2": "bus#n#1"})

headers = {
      "User-Agent" : "pyclient",
}

request = urllib2.Request("http://127.0.0.1:8080/api", data, headers)
response = urllib2.urlopen(request)
print response.read().split(':')[1][0:-1]
