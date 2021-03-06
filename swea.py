import numpy as np
import pylab as plt
import spiceypy
import celsius
from . import sdc_interface
from functools import wraps
from matplotlib.colors import LogNorm

import cdflib

import os

def load_swea_l2_summary(start, finish, kind='svyspec', http_manager=None,
        delete_others=True, cleanup=False, verbose=None):
    kind = kind.lower()

    if not delete_others:
        raise RuntimeError("Not written yet")

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
                    'swe/l2/%04d/%02d/mvn_swe_l2_%s_*_v*_r*.cdf' % \
                                            (year, month, kind),
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

    # Check for duplicates:
    if len(files) != len(set(files)):
        raise ValueError("Duplicates appeared in files to load: " + ", ".join(files))

    if cleanup:
        print('SWEA L2 cleanup complete')
        return

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)

    if not files:
        return dict()

    if kind == 'svyspec':
        output = {'time':None, 'def':None}
        for f in sorted(files):
            c = cdflib.CDF(f)

            if output['time'] is None:
                output['time'] = c['time_unix']
                output['def']  = c['diff_en_fluxes'].T

                # Some weird formatting here:
                output['energy']  = np.array(
                    [c['energy'][i] for i in range(c['energy'].shape[0])]
                )
                output['energy'] = output['energy'][::-1]
            else:
                output['time'] = np.hstack((output['time'],
                                    c['time_unix']))
                output['def'] = np.hstack((output['def'],
                                    c['diff_en_fluxes'].T))

                if output['energy'].shape != c['energy'].shape:
                    raise ValueError("Energy range has changed!")

            c.close()
        output['def'] = output['def'][::-1,:]
    else:
        raise ValueError("Input kind='%s' not recognized" % kind)

    output['time'] = output['time'] + celsius.spiceet("1970-01-01T00:00")

    return output

def plot_swea_l2_summary(swea_data, max_times=4096, cmap=None, norm=None,
        labels=True, ax=None, colorbar=True):

    energy_range = (2, 4000.)
    def_range    = (1e6, 1e9)

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if cmap is None: cmap = 'Spectral_r'
    if norm is None: norm = LogNorm(def_range[0], def_range[1])

    if not 'def' in swea_data:
        print('Data is empty?')
        # extent = (t[0], t[-1], swea_data['energy'][0], swea_data['energy'][-1])
        d = np.empty(4).reshape((2,2)) + np.nan
        t = plt.xlim()
    else:
        d = swea_data['def']
        t = swea_data['time']

        if d.shape[1] > max_times:
            n = int(np.floor(d.shape[1] / max_times))
            d = d[:,::n]
            t = t[::n]

        energy_range = (swea_data['energy'][0], swea_data['energy'][-1])

    extent = (t[0], t[-1], energy_range[0], energy_range[1])

    img = plt.pcolormesh(t, swea_data['energy'], d,
        norm=norm, cmap=cmap
    )
    # plt.xlim(t[0], t[-1])

    plt.yscale('log')
    plt.ylim(energy_range[0], energy_range[1])

    if labels:
        plt.ylabel("E / eV")

    if colorbar:
        plt.colorbar(cax=celsius.make_colorbar_cax()).set_label('SWEA D.E.F.')

    return img

def cleanup(start=None, finish=None):
    if not start: start = celsius.spiceet("2014-09-22T00:00")
    if not finish: finish = celsius.now()

    # Cleanup commands
    load_swea_l2_summary(start, finish, cleanup=True, verbose=True)

if __name__ == '__main__':
    plt.close('all')
    t0 = celsius.spiceet("2015-01-08")
    t1 = t0 + 86400. * 2. + 1

    d = load_swea_l2_summary(t0, t1)
    plot_swea_l2_summary(d)
    celsius.setup_time_axis()

    plt.show()
