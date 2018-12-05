"""Interface for the MAVEN SDC at LASP.
"""

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
    """Extract all links from an HTML page"""
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
    def __init__(self, remote_path, username, password, local_path, update_interval=86400., verbose=False, silent=False):
        """Create a HTTP_Manager
Args:
    remote_path: base URL to copy from
    username: user name to authenticate with, if required
    password: password for the user
    local_path: base URL to copy to
    update_interval: time in seconds before an index.html file is considered
        out of date
    verbose: print out extra info
    silent: print nothing.

Returns:
    Instance.

Example:

"""
        self.remote_path = remote_path
        self.username = username
        self.password = password
        self.local_path = local_path
        self.update_interval = update_interval
        self.verbose = verbose
        self.version = True
        self.download = True
        self.silent = silent

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

    def query(self, query, version_function=None, date_function=None,
                start=None, finish=None, cleanup=False, verbose=None,
                silent=None):
        """Takes a query, returns a list of local files that match.

Will first query the remote server, download missing files, and then delete local files that match the query but are no longer present on the remote. The implicit assumption here is that the remote directory is PERFECTLY maintained.

Args:
    query: query string with wildcards to locate a file on the remote server,
        e.g. 'sci/lpw/l2/2015/01/mvn_lpw_l2_lpnt_*_v*_r*.cdf'

    version_function: takes the expanded wildcards from the query, and converts
        them to a number used to compare versions and releases (higher=better).
        For example:
            lambda x: (x[0], float(x[1]) + float(x[2])/100.)
        to generate 1.02 for V1, R2 for the above query (2nd and 3rd wildcards)

    date_function: takes the expanded wildcards from the query, and converts to
        a date for the content of the file, for example:
            lambda x: yyyymmdd_to_spiceet(x[0])
        for the above query example.

    start: start date SPICEET, ignored if not set
    finish: finish date, ignored if not set. 'finish' must be set if 'start' is
        (can use np.inf, if you want)

Returns: List of local files, freshly downloaded if necessary, that satisfy the
    query supplied.
        """

        file_list = []
        split_query = query.split('/')
        query_base_path = '/'.join(split_query[:-1]) + '/'
        query_filename  = split_query[-1]

        if verbose is None: verbose = self.verbose
        if silent is None: silent = self.silent

        if version_function is None:
            version_function = lambda x: 0, ''.join(x)

        self.current_re = re.compile(query_filename.replace("*", "(\w*)"))

        if not os.path.exists(self.local_path + query_base_path):
            os.makedirs(self.local_path + query_base_path)

        check_time = False
        if start or finish: check_time = True

        if check_time and (date_function is None):
            raise ValueError("Start and finish are set, but date_function is not")

        if check_time:
            start_day = celsius.spiceet(celsius.utcstr(start, 'ISOC')[:10])
            finish_day = celsius.spiceet(celsius.utcstr(finish, 'ISOC')[:10]) \
                                + 86398. #1 day - 1s - 1 (possible) leap second

        # if verbose:
        #   print 'Remote path: ', self.remote_path + query_base_path

        ok_files = {}  # key will be the unique id of the file, value will be (version, the full name, local == True)
        files_to_delete = []

        n_downloaded = 0
        n_deleted    = 0

        # Find local matches
        for f in os.listdir(self.local_path + query_base_path):
            tmp = self.current_re.match(f)
            if tmp:
                unique_id, version_number = version_function(tmp.groups())

                if check_time:
                    file_time = date_function(tmp.groups())
                    if (file_time < start_day) or (file_time > finish_day):
                        continue

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
                age = py_time.time() - os.path.getmtime(index_path)
                if age < self.update_interval:
                    update_index = False

            if update_index:
                try:
                    self._get_remote(remote_path, index_path)
                except IOError as e:
                    if verbose:
                        print('Index %s does not exist' % remote_path)

                    if ok_files:
                        raise RuntimeError("""No remote index available, but local matches were found anyway. This should never happen.

                        Details:
                            remote_path: %s
                            index_path: %s
                        """.format(remote_path, index_path))

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
                        if (file_time < start_day) or (file_time > finish_day):
                            continue

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
                        try:
                            self._get_remote(fname,
                                self.local_path + query_base_path + f[1])
                        except IOError as e:
                            print('Error encountered - index may be out of date?')
                            raise

                        # Update the name with the local directory
                        ok_files[k] = (f[0],
                            self.local_path + query_base_path + f[1],f[2])
                        n_downloaded += 1

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
            n_deleted += 1

        if not silent:
            print('Query %s: Returning %d (DL: %d, DEL: %d)' %
                (query, len(ok_files), n_downloaded, n_deleted))
        return [f[1] for f in list(ok_files.values())]



if __name__ == '__main__':
    pass
# maven_http_manager = HTTP_Manager(
#         'http://sprg.ssl.berkeley.edu/data/maven/data/sci/',
#         os.getenv('MAVENPFP_USER_PASS').split(':')[0],
#         os.getenv('MAVENPFP_USER_PASS').split(':')[1],
#         os.getenv('MAVEN_DATA_DIR', os.getenv('SC_DATA_DIR')+'maven/'),
#         verbose=False)
