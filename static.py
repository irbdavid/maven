import numpy as np
import pylab as plt
import spiceypy
import celsius
from . import sdc_interface
from functools import wraps
from matplotlib.colors import LogNorm

import cdflib

import os

STATIC_PRODUCTS = {
    'c0':'c0-64e2m',
    'c6':'c6-32e64m',
    'c8':'c8-32e16d',
}

def load_static_l2(start, finish, kind='c0',
        http_manager=None, delete_others=True, cleanup=False, verbose=None):
    kind = kind.lower()

    full_kind = STATIC_PRODUCTS[kind]

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
                    'sta/l2/%04d/%02d/mvn_sta_l2_%s_*_v*_r*.cdf' % \
                                            (year, month, full_kind),
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
        print('static L2 Cleanup complete')
        return

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)
    #
    # if kind == 'c6':
    #     output = {'time':None, 'eflux':None, 'static_kind':'c6'}
    #     for f in sorted(files):
    #         c = pycdf.CDF(f)
    #
    #         if output['time'] is None:
    #             output['time'] = c.varget('time_unix'])
    #             output['eflux']  = c.varget('eflux']).T
    #
    #             output['energy']  = c.varget('energy'][0,:,0])
    #             output['mass']  = c.varget('mass_arr'][:,0,0])
    #
    #         else:
    #             output['time'] = np.hstack((output['time'],
    #                                 c.varget('time_unix'])))
    #             output['eflux']  = np.hstack((output['time'],
    #                                 c.varget('eflux'].T)))
    #
    #             # if output['energy'].shape != c['energy'].shape[1]:
    #             #     raise ValueError("Energy range has changed!")
    #             #
    #             # if output['mass'].shape != c['mass_arr'].shape[0]:
    #             #     raise ValueError("Mass range has changed!")
    #
    #         c.close()

    if kind == 'c0':
        t0 = celsius.spiceet("1970-01-01T00:00")
        output = {'blocks':[], 'static_kind':'c0'}
        for f in sorted(files):
            c = cdflib.CDF(f)

            data = c['eflux']
            last_ind = None
            last_block_start = None
            N = data.shape[0]
            # print(c['eflux'].shape, c['energy'].shape, c['time_unix'].shape)
            output['blocks'].append([
                        c['time_unix'],
                        c['energy'], c['eflux']])
            # for i in range(data.shape[0]):
            #
            #     if last_ind is None:
            #         last_ind = c['swp_ind'][i]
            #         last_block_start = i
            #
            #     if (c['swp_ind'][i] != last_ind) or (i == N):
            #
            #         img = data[last_block_start:i-1, :, :].sum(axis=1)
            #         extent = (
            #                 c['time_unix'][last_block_start] + t0,
            #                 c['time_unix'][i-1] + t0,
            #                 c['energy'][0, -1, last_ind],
            #                 c['energy'][0, 0, last_ind],
            #             )
            #         # print(img.shape, c['energy'].shape, c['time_unix'].shape)
            #         # print(last_ind, extent)
            #         # output['blocks'].append((extent, (img.T[::-1,:])))
            #         output['blocks'].append( (img.T[::-1,:]) )
            #         # plt.imshow(np.log10(img.T[::-1,:]), extent=extent,
            #         #             origin='lower', interpolation='nearest')
            #         last_ind = None

            c.close()


    else:
        raise ValueError("Input kind='%s' not recognized" % kind)

    # output['time'] = output['time'] + celsius.spiceet("1970-01-01T00:00")

    return output

def plot_static_l2_summary(static_data, plot_type='Energy',
        max_times=4096, cmap=None, norm=None,
        labels=True, ax=None, colorbar=True):

    if not 'static_kind' in static_data:
        raise ValueError("Data supplied not from static?")

    if not static_data['static_kind'] == 'c0':
        raise ValueError("I only know about C0, for now")

    if cmap is None: cmap = 'Spectral_r'
    if norm is None: norm = LogNorm(1e3, 1e9)

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    imgs = []

    x0, x1 = plt.xlim()

    for time, energy, data in static_data['blocks']:
        if extent[-1] < x0: continue
        if extent[0] > x1: continue

        img = plt.pcolormesh(
            data, extent=extent, interpolation='nearest', origin='lower',
            norm=norm, cmap=cmap, aspect='auto'
        )
        imgs.append(img)
    plt.yscale('log')
    # plt.xlim(t0, t1)
    plt.ylim(extent[2], extent[3])

    if labels:
        plt.ylabel("E / eV")

    if colorbar:
        plt.colorbar(cax=celsius.make_colorbar_cax()).set_label('static D.E.F.')

    return imgs

def cleanup(start=None, finish=None):
    if not start: start = celsius.spiceet("2014-09-22T00:00")
    if not finish: finish = celsius.now()

    # Cleanup commands
    load_static_l2_summary(start, finish, cleanup=True)

if __name__ == '__main__':
    plt.close('all')
    t0 = celsius.spiceet("2015-01-08")
    t1 = t0 + 86400. * 2. + 1
    t1 = t0 + 86400. - 1.

    # d = load_static_l2_summary(t0, t1, kind='onboardsvyspec')
    # plot_static_l2_summary(d)

    d = load_static_l2_summary(t0, t1, kind='c0')
    plot_static_l2_summary(d)
    # plt.subplot(211)
    # plt.plot(d['time'], d['density'])
    # plt.subplot(212)
    # plt.plot(d['time'], d['velocity'][0], 'r.')
    # plt.plot(d['time'], d['velocity'][1], 'g.')
    # plt.plot(d['time'], d['velocity'][2], 'b.')
    #
    celsius.setup_time_axis()

    plt.show()
