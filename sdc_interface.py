import celsius
import numpy as np

import logging
import urllib.request, urllib.error, urllib.parse
import tempfile
import os

import time as py_time
from html.parser import HTMLParser
import re

from spacepy import pycdf

def yyyymmdd_to_spiceet(x):
    return celsius.spiceet(x[:4] + '-' + x[4:6] + '-' + x[6:8] + 'T00:00')

def merge_attrs(dict_out, name, dict_in, input_name=None, transpose=False):
    if input_name is None:
        input_name = name
    v = np.array(dict_in[input_name])
    if transpose:
        v = v.T

    if dict_out[name] is None:
        dict_out[name] = v
    else:
        dict_out[name] = np.hstack( (dict_out[name], v) )

class IndexParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.files = []

    def extract_links(self, f):
        self.files = []
        self.feed(f)
        return self.files

    def handle_starttag(self, tag, attrs):
        if tag == "a":
           for name, value in attrs:
               if name == "href":
                   self.files.append(value)
                   # return value

class HTTP_Manager(object):
    """Used to transfer files via HTTP, following Berkeley's system"""

    def __init__(self, remote_path, username, password, local_path, update_interval=1, verbose=False):
        """Setup"""
        self.remote_path = remote_path
        self.username = username
        self.password = password
        self.local_path = local_path
        self.update_interval = update_interval
        self.verbose = verbose
        self.version = True
        self.download = True

        self.passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        self.passman.add_password(None, remote_path, username, password)
        self.authhandler = urllib.request.HTTPBasicAuthHandler(self.passman)
        self.opener = urllib.request.build_opener(self.authhandler)
        urllib.request.install_opener(self.opener)

        self.index_parser = IndexParser()

    def _get_remote(self, url, local):

        try:
            pagehandle = urllib.request.urlopen(url)
            if self.verbose:
                print('Fetching %s ... ' % url)
            thepage = pagehandle.read()
        except urllib.error.HTTPError as e:
            print(e)
            raise IOError('Could not read %s' % url)

        with open(local, 'wb') as f:
            f.write(thepage)

        # if self.verbose: print 'Wrote %s to %s' % (url, local)

        return local

    def query(self,  query, version_function=None, date_function=None,
                start=None, finish=None, cleanup=False, verbose=None):
        """Takes a query, returns a list of local files that match.  If set, will first query the remote
        server, download missing files, and then delete local files that match the query but are no longer
        present on the remote.

        The implicit assumption here is that the remote directory is PERFECTLY maintained.

        :query: e.g. 'sci/lpw/l2/2015/01/mvn_lpw_l2_lpnt_*_v*_r*.cdf'
        :version_function: takes the matched wildcards from the query, and converts them to a number used to compare
            versions (higher = better)
        :date_function: takes the matched wildcards from the query, and converts to a date for the content of the file
        :start: start date, ignored if not set
        :finish: finsh date, ignored if not set. finish must be set if start is (use np.inf, if you want)
        :returns: List of local files, freshly downloaded if necessary, that satisfy the query

        """

        file_list = []
        split_query = query.split('/')
        query_base_path = '/'.join(split_query[:-1]) + '/'
        query_filename  = split_query[-1]

        if verbose is None: verbose = self.verbose

        if version_function is None:
            version_function = lambda x: 0, ''.join(x)

        self.current_re = re.compile(query_filename.replace("*", "(\w*)"))

        if not os.path.exists(self.local_path + query_base_path):
            os.makedirs(self.local_path + query_base_path)

        check_time = False
        if start or finish: check_time = True

        if check_time and (date_function is None):
            raise ValueError("Start and finish are set, but date_function is not")

        # if verbose:
        #   print 'Remote path: ', self.remote_path + query_base_path

        ok_files = {}  # key will be the unique id of the file, value will be (version, the full name, local == True)
        files_to_delete = []

        # Find local matches
        for f in os.listdir(self.local_path + query_base_path):
            tmp = self.current_re.match(f)
            if tmp:
                unique_id, version_number = version_function(tmp.groups())

                if check_time:
                    file_time = date_function(tmp.groups())
                    if (file_time < start) or (file_time > finish): continue

                if unique_id in ok_files:
                    if ok_files[unique_id][0] < version_number:
                        ok_files[unique_id] = (version_number, self.local_path + query_base_path + f, True)
                else:
                    ok_files[unique_id] = (version_number, self.local_path + query_base_path + f, True)

        if verbose:
            if ok_files:
                print('%d local matches with highest version %f' % (len(ok_files), max([v[0] for v in list(ok_files.values())])))
            else:
                print('No local matches')

        # Find remote matches
        if self.download:
            index_path = self.local_path + query_base_path + '.remote_index.html'
            remote_path = self.remote_path + query_base_path

            update_index = True
            if os.path.exists(index_path):
                age = os.path.getmtime(index_path) - py_time.time()
                if age < self.update_interval: update_index = False

            if update_index:
                try:
                    self._get_remote(remote_path, index_path)
                except IOError as e:
                    if verbose:
                        print('Index %s does not exist' % remote_path)

                    if ok_files:
                        raise RuntimeError("No remote index available, but local matches were found anyway. This should never happen.")

                    return []


            with open(index_path) as f:
                remote_files = self.index_parser.extract_links(f.read()) # without the remote + base path

            if not remote_files:
                raise IOError('No remote files found from index file')

            # inspect each file, remove if it doesn't match the query, or is not the most recent version
            for f in remote_files:
                tmp = self.current_re.match(f)
                if tmp:
                    unique_id, version_number = version_function(tmp.groups())

                    if check_time:
                        file_time = date_function(tmp.groups())
                        if (file_time < start) or (file_time > finish): continue

                    if unique_id in ok_files:
                        if ok_files[unique_id][0] < version_number:
                            # if we are overwriting a local entry, we will also need to delete the original file
                            if ok_files[unique_id][2]:
                                files_to_delete.append(ok_files[unique_id][1])

                            ok_files[unique_id] = (version_number, f, False)

                    else:
                        ok_files[unique_id] = (version_number, f, False)


            if not cleanup:
                for k in list(ok_files.keys()):
                    f = ok_files[k]
                    fname = self.remote_path + query_base_path + f[1]
                    if not f[2]: # download remote file
                        self._get_remote(fname,
                            self.local_path + query_base_path + f[1])

                        # Update the name with the local directory
                        ok_files[k] = (f[0],
                                self.local_path + query_base_path + f[1],
                                f[2])

            if verbose:
                if ok_files:
                    print('%d remote matches with highest version %f' % \
                        (len(ok_files), max([v[0] for v in list(ok_files.values())])))
                else:
                    print('No remote matches')

        for f in files_to_delete:
            if verbose:
                print('Deleting ' + f)
            os.remove(f)

        return [f[1] for f in list(ok_files.values())]

maven_http_manager = HTTP_Manager(
        'http://sprg.ssl.berkeley.edu/data/maven/data/sci/',
        os.getenv('MAVENPFP_USER_PASS').split(':')[0],
        os.getenv('MAVENPFP_USER_PASS').split(':')[1],
        os.getenv('SC_DATA_DIR') + 'maven/spg/data/maven/data/sci/',
        verbose=True)

if __name__ == '__main__':
    pass
