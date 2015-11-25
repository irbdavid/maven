import numpy as np
import pylab as plt
import os

import spiceypy
import celsius
from matplotlib.colors import LogNorm

from .kp_shortname_lookup import kp_shortname_lookup
from . import sdc_interface


from spacepy import pycdf

class MavenKPData(dict):
    """A helper class that organises the KP data dictionary, allowing getting/setting via attributes as well as dictionary keys."""
    def __init__(self, *args, **kwargs):
        super(MavenKPData, self).__init__(*args, **kwargs)
        self.__dict__ = self

class MavenKPDataDescription(object):
    """Stores longer information for each KP item"""
    def __init__(self, name=None, units=None, fmt=None, notes=None):
        super(MavenKPDataDescription, self).__init__()
        self.name = name
        self.units = units
        self.fmt = fmt
        self.notes = notes

    def plot(self, val):
        pass


def read_kp_shortnames(filename=None):
    """Parse the accompanying list of short names.
    This relies on having the columns in the file synced with those in the data.  This seems to be the case. """

    output = {}
    output[0] = 'spacecraft', 'time'

    if filename:
        with open(filename) as f:
            lines = f.read().splitlines()
    else:
        lines = kp_shortname_lookup.splitlines()

    for line in lines:
        if line[0] == '#': continue
        if line == '': continue
        var, inx = line.split('=')
        if not '.' in var: continue
        inst, shortname = var.strip().split('.')
        inx = int(inx.split('[')[1][:-1])
        output[inx] = (inst, shortname)
        # print inx, inst, shortname
    return output

def kp_read_files(files):
    """Read a list of ASCII KP files, returning data and decscription objects."""

    data = MavenKPData()
    descriptions = MavenKPData()

    shortnames = read_kp_shortnames()

    for inx in shortnames:
        inst, name = shortnames[inx]
        if not inst in data:
            data[inst] = MavenKPData()
            descriptions[inst] = MavenKPData()
        data[inst][name] = None
        descriptions[inst][name] = MavenKPDataDescription()

    converters = {}
    converters[0] = celsius.spiceet
    converters[210] = lambda x: 1 if x=='I' else 0

    def parse_float(x):
        try:
            return float(x)
        except ValueError:
            return np.nan

    for i in range(1, 220):
        if not i in converters:
            converters[i] = parse_float

    first_file = True
    for filename in sorted(files):

        # Read information from the header, compare it to the shortnames
        # Populate the description object
        if first_file:
            with open(filename) as f:
                while True:
                    line = f.readline()
                    if 'PARAMETER' in line:
                        f.readline()
                        break

                i = 0
                last_non_quality = None

                while True:
                    line = f.readline()
                    if line.strip() == '#': continue
                    if line == '': continue
                    name = line[1:60].strip()
                    inst = line[60:72].strip()
                    units = line[72:90].strip()
                    fmt  = line[100:122].strip()
                    notes = line[122:].strip()
                    if 'Quality' in name:
                        name = last_non_quality + ' Quality'
                    else:
                        last_non_quality = name
                        # print name[:10], inst

                    # print inst, name, i
                    if i in shortnames:
                        inst, name = shortnames[i]
                        descriptions[inst][name].name = name
                        descriptions[inst][name].units = units
                        descriptions[inst][name].fmt = fmt
                        descriptions[inst][name].notes = notes
                    else:
                        print("Missing %s at column %d" % (name, i))

                    i += 1
                    if i > 210: break

        # Now NP gets to read the thing:
        tmp = np.loadtxt(filename, converters=converters).T

        if first_file:
            for inx in shortnames:
                inst, name = shortnames[inx]
                data[inst][name] = tmp[inx]
            first_file = False
        else:
            for inx in shortnames:
                inst, name = shortnames[inx]
                data[inst][name] = np.hstack((data[inst][name], tmp[inx]))

    data['time'] = data['spacecraft']['time'] # a link
    descriptions['time'] = descriptions['spacecraft']['time'] # a link
    data['descriptions'] = descriptions

    return data

def load_kp_data(start, finish, vars='ALL', truncate_time=True,
        http_manager=None, cleanup=False, verbose=None):
    """Reads MAVEN kp data into a structure.  Downloads / syncs if neccessary.
    Args:
        start, finish: SPICEET times
        vars: variable names to store (not implemented - default ALL)
        http_manager: connection to use
        cleanup: if True, no data will be downloaded or returned, and instead
            only superceded local files will be deleted
        verbose: locally overrides http_manager.verbose
        truncate_time: slice out only those points between start and finish,
            or return whole days if false

    Returns:
        results of kp_read_files for the located files
    """

    if http_manager is None:
        http_manager = sdc_interface.maven_http_manager

    t = start
    year, month = celsius.utcstr(t,'ISOC').split('-')[0:2]
    year = int(year)
    month = int(month)
    #  Each month:
    files = []
    while t < finish:
        files.extend(
                http_manager.query(
                    'kp/insitu/%04d/%02d/mvn_kp_insitu_*_v*_r*.tab' % \
                                            (year, month),
                    start=start, finish=finish,
                    version_function=\
                        lambda x: (x[0], float(x[1]) + float(x[2])/100.),
                    date_function=lambda x: sdc_interface.yyyymmdd_to_spiceet(x[0]),
                    cleanup=cleanup, verbose=verbose
                )
            )
        month += 1
        if month > 12:
            month = 0o1
            year += 1
        t = celsius.spiceet('%d-%02d-01T00:00' % (year, month))

    if cleanup:
        print('KP cleanup complete')
        return

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)

    if not files:
        raise IOError("No KP data found")

    data = kp_read_files(files)

    if truncate_time:
        inx, = np.where((data.time > start) & (data.time < finish))
        for k in list(data.keys()):
            if k not in ('time', 'descriptions'):
                for kk in list(data[k].keys()):
                    data[k][kk] = data[k][kk][inx]

    data['time'] = data['spacecraft']['time'] # a link

    return data

def cleanup(start=None, finish=None):
    if not start: start = celsius.spiceet("2014-09-22T00:00")
    if not finish: finish = celsius.now()

    # Cleanup commands
    load_kp_data(start, finish, cleanup=True, verbose=True)

if __name__ == '__main__':
    plt.close('all')
    # d = kp_test()

    t0 = celsius.spiceet("2015-01-08")
    t0 = celsius.spiceet("2015-05-01")
    t1 = t0 + 86400. * 10. - 1
    # t1 = t0 + 86400. - 1.

    data = load_kp_data(t0, t1)
    celsius.setup_time_axis()

    plt.plot(data.time, data.swia.hplus_density, 'k-')
    # plt.plot(data.time, data.swea.solarwind_e_density, 'r-')
    plt.plot(data.time, data.static.oplus_density, 'b-')
    plt.plot(data.time, data.static.o2plus_density, 'c-')
    # plt.plot(data.time, data.ngims.o2plus_density, 'c:x')
    plt.plot(data.time, data.lpw.electron_density, 'g-')

    # plt.plot(data.time, data.swia.solarwind_dynamic_pressure, 'k-')
    # plt.plot(data.time, data.lpw.electron_density *
    #         data.lpw.electron_temperature * 1.602e-4, 'k-')

    plt.yscale('log')

    plt.show()
