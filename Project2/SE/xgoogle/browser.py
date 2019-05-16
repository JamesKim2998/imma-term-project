#!/usr/bin/python
# -*- coding: utf8 -*-
#
# Peteris Krumins (peter@catonmat.net)
# http://www.catonmat.net  --  good coders code, great reuse
#
# http://www.catonmat.net/blog/python-library-for-google-search/
#
# Code is licensed under MIT license.
#

import ssl
import random
import socket
import urllib
import urllib.request
import http.client
import http.cookiejar
import http.cookies

BROWSERS = (
    # Top most popular browsers in my access.log on 2009.02.12
    # tail -50000 access.log |
    #  awk -F\" '{B[$6]++} END { for (b in B) { print B[b] ": " b } }' |
    #  sort -rn |
    #  head -20
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:2.0) Gecko/20100101 Firefox/4.0',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.6) Gecko/2009011913 Firefox/3.0.6',
    'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.5; en-US; rv:1.9.0.6) Gecko/2009011912 Firefox/3.0.6',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.6) Gecko/2009011913 Firefox/3.0.6 (.NET CLR 3.5.30729)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.6) Gecko/2009020911 Ubuntu/8.10 (intrepid) Firefox/3.0.6',
    'Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US; rv:1.9.0.6) Gecko/2009011913 Firefox/3.0.6',
    'Mozilla/5.0 (Windows; U; Windows NT 6.0; en-US; rv:1.9.0.6) Gecko/2009011913 Firefox/3.0.6 (.NET CLR 3.5.30729)',
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/525.19 (KHTML, like Gecko) Chrome/1.0.154.48 Safari/525.19',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727; .NET CLR 3.0.04506.30; .NET CLR 3.0.04506.648)',
    'Mozilla/5.0 (X11; U; Linux x86_64; en-US; rv:1.9.0.6) Gecko/2009020911 Ubuntu/8.10 (intrepid) Firefox/3.0.6',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.5) Gecko/2008121621 Ubuntu/8.04 (hardy) Firefox/3.0.5',
    'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_5_6; en-us) AppleWebKit/525.27.1 (KHTML, like Gecko) Version/3.2.1 Safari/525.27.1',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.115 Safari/537.36',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322)',
    'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
)

TIMEOUT = 5  # socket timeout

class BrowserError(Exception):
    def __init__(self, url, error):
        self.url = url
        self.error = error

class PoolHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        """Connect to the host and port specified in __init__."""
        msg = "getaddrinfo returns an empty list"
        for res in socket.getaddrinfo(self.host, self.port, 0,
                                      socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            try:
                self.sock = socket.socket(af, socktype, proto)
                if self.debuglevel > 0:
                    print("connect: (%s, %s)" % (self.host, self.port))
                self.sock.settimeout(TIMEOUT)
                self.sock.connect(sa)
            except socket.error as msg:
                if self.debuglevel > 0:
                    print('connect fail:', (self.host, self.port))
                if self.sock:
                    self.sock.close()
                self.sock = None
                continue
            break
        if not self.sock:
            raise socket.error(msg)

class PoolHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(PoolHTTPConnection, req)

class Browser(object):
    """Provide a simulated browser object.
    """
    def __init__(self, user_agent=BROWSERS[0], debug=False, use_pool=False):
        self.headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-GB,en-US;q=0.8,en;q=0.6',
            # 'Accept-Encoding': 'deflate'
        }
        self.debug = debug
        self._cj = http.cookiejar.CookieJar()

        self.handlers = [PoolHTTPHandler]
        self.handlers.append(urllib.request.HTTPCookieProcessor(self._cj))

        self.opener = urllib.request.build_opener(*self.handlers)
        self.opener.addheaders = []

        ssl._create_default_https_context = ssl._create_unverified_context

        try:
            conn = self.opener.open("http://www.google.com/ncr")
            conn.info()  # retrieve session cookie
        except Exception as e:
            print(e)

    def get_page(self, url, data=None):
        # handlers = [PoolHTTPHandler]
        # opener = urllib.request.build_opener(*handlers)
        if data: data = urllib.urlencode(data)
        request = urllib.request.Request(url, data, self.headers)
        try:
            response = self.opener.open(request)
            return response.read()
        except (urllib.error.HTTPError) as e:
            # Check if we've reached the captcha
            if e.code == 503:
                print("Error: Captcha page has been reached, exiting...")
                exit(1)
            raise BrowserError(url, str(e))
        except (urllib.error.URLError) as e:
            raise BrowserError(url, str(e))
        except (socket.error, socket.sslerror) as msg:
            raise BrowserError(url, msg)
        except socket.timeout as e:
            raise BrowserError(url, "timeout")
        except KeyboardInterrupt:
            raise
        except:
            raise BrowserError(url, "unknown error")

    def set_random_user_agent(self):
        self.headers['User-Agent'] = random.choice(BROWSERS)
        return self.headers['User-Agent']

    def get_user_agent(self):
        return self.headers['User-Agent']
