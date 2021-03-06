import numpy as np
import pylab as plt
import spiceypy
import celsius

from maven import sdc_interface
import cdflib
# CDFConverter = sdc_interface.CDFConverter

from functools import wraps
from matplotlib.colors import LogNorm

# from spacepy import pycdf

import os
from scipy.io.idl import readsav


def get_densities(start, finish=None, verbose=False, sweeps=True,
                cleanup=False):
    """Routine to extract Dave's own processed densities.  Ignore / don't use."""
    raise RunTimeError("Not for use.")

    if finish is None: finish = start + 86400. - 1.

    if start > finish: raise ValueError("Start %f exceeds %f" % (start, finish))

    directory = os.getenv("SC_DATA_DIR", os.path.expanduser("~/data/"))
    directory += 'maven/ping/'

    t = start
    chunks = []
    while t < finish:
        try:
            date = celsius.utcstr(t, 'ISOC')[:10]
            fname = directory + date[:4] + '/' + date + '.sav'
            tmp = readsav(fname)
            if not 'sza' in tmp:
                print(fname + ' out of date, skipping')
                t+=86400.
                continue

            n = len(tmp['time'])
            for k in ('sza', 'density', 'flag'):
                if tmp[k].shape[-1] != n:
                    print('Malformed ', fname)
                    continue


            chunks.append(tmp)
            if verbose:
                print(fname + ', ' +  str(len(chunks[-1]['time'])))

        except IOError as e:
            if verbose:
                print("Missing: " + fname)

        t += 86400.

    if not chunks:
        print('No data found')
        return chunks

    banned_keys = ('probe', 'spec', 'spec_f', 'iv1', 'iv2') # The spectra are not retained

    output = {}
    for k in list(chunks[0].keys()):
        # print k, chunks[0][k].shape
        if k in banned_keys: continue
        output[k] = np.hstack([c[k] for c in chunks])
    print(output['sza'].shape == output['time'].shape)

    if sweeps:
        for k in ('iv1', 'iv2'):
            output[k] = {}
            for kk in chunks[0][k].dtype.names:
                if kk.lower() in banned_keys: continue
                output[k][kk.lower()] = np.hstack([c[k][kk][0] for c in chunks])

    print(output['sza'].shape == output['time'].shape)
    inx, = np.where((output['time'] > start) & (output['time'] < finish))
    for k in list(output.keys()):
        if k in banned_keys: continue
        output[k] = output[k][...,inx]

    inx = np.argsort(output['time'])
    for k in list(output.keys()):
        if k in banned_keys: continue
        output[k] = output[k][...,inx]

    if sweeps:
        for k in ('iv1', 'iv2'):
            inx, = np.where((output[k]['time'] > start) & (output[k]['time'] < finish))
            for kk in list(output[k].keys()):
                output[k][kk] = output[k][kk][inx]
            inx = np.argsort(output[k]['time'])
            for kk in list(output[k].keys()):
                output[k][kk] = output[k][kk][inx]

    if cleanup:
        print("Cleaning up some timing error - any negative time steps being erased")
        dt = np.diff(output['time'])
        inx, = np.where(dt < 0.)
        if inx.size != 0:
            for k in list(output.keys()):
                if k in banned_keys: continue
                output[k][inx+1] *= np.nan

        if sweeps:
            for k in ('iv1', 'iv2'):
                inx, = np.where(np.diff(output[k]['time']) < 0.)
                if inx.size != 0:
                   for kk in list(output[k].keys()):
                        output[k][kk][inx + 1] *= np.nan

    return output

def spice_wrapper(n=1):
    """Wrapper around spiceypy.spkpos that handles array inputs, and provides useful defaults"""
    def actual_decorator(f):
        def g(t):
            try:
                return f(t)
            except spiceypy.SpiceException:
                return np.repeat(np.nan, n)

        @wraps(f)
        def inner(time):
            if hasattr(time, '__iter__'):
                return np.vstack([g(t) for t in time]).T
            else:
                return g(time)
        return inner
    return actual_decorator

@spice_wrapper(n=3)
def ram_angles(time):
    """Return the angle between SC-Y and the ram direction (0. = Y to ram)"""
    p = spiceypy.spkezr('MAVEN', time, 'IAU_MARS', 'NONE', 'MARS')[0][3:]
    r = spiceypy.pxform('IAU_MARS', 'MAVEN_SPACECRAFT', time)
    a = spiceypy.mxv(r, p)

    e = np.arctan( np.sqrt(a[1]**2. + a[2]**2.) / a[0]) * 180./np.pi
    f = np.arctan( np.sqrt(a[0]**2. + a[2]**2.) / a[1]) * 180./np.pi
    g = np.arctan( np.sqrt(a[0]**2. + a[1]**2.) / a[2]) * 180./np.pi
    if e < 0.: e = e + 180.
    if f < 0.: f = f + 180.
    if g < 0.: g = g + 180.

    return np.array((e,f,g))

@spice_wrapper(n=3)
def sun_angles(time):
    """Return the angle between SC-Y and the sun direction (0. = Y to sun)"""
    a = spiceypy.spkpos('SUN', time, 'MAVEN_SPACECRAFT', 'NONE', 'MAVEN')[0]

    e = np.arctan( np.sqrt(a[1]**2. + a[2]**2.) / a[0]) * 180./np.pi
    f = np.arctan( np.sqrt(a[0]**2. + a[2]**2.) / a[1]) * 180./np.pi
    g = np.arctan( np.sqrt(a[0]**2. + a[1]**2.) / a[2]) * 180./np.pi
    if e < 0.: e = e + 180.
    if f < 0.: f = f + 180.
    if g < 0.: g = g + 180.

    return np.array((e,f,g))


def lpw_l2_load(start, finish, kind='lpnt', http_manager=None, cleanup=False,
                    verbose=None):
    """Finds and loads LPW L2 data"""

    if http_manager is None: http_manager = sdc_interface.maven_http_manager
    kind = kind.lower()

    t = start
    year, month = celsius.utcstr(t,'ISOC').split('-')[0:2]
    year = int(year)
    month = int(month)
    #  Each month:
    files = []
    while t < finish:
        # print year, month
        files.extend(
                http_manager.query(
                    'lpw/l2/%04d/%02d/mvn_lpw_l2_%s_*_v*_r*.cdf' % \
                                            (year, month, kind),
                    start=start, finish=finish,
                    version_function=\
                        lambda x: (x[0], float(x[1]) + float(x[2])/100.),
                    date_function=lambda x:
                                sdc_interface.yyyymmdd_to_spiceet(x[0]),
                    verbose=verbose
                )
            )
        month += 1
        if month > 12:
            month = 1
            year += 1
        t = celsius.spiceet('%d-%02d-01T00:00' % (year, month))

    # Check for duplicates:
    if len(files) != len(set(files)):
        raise ValueError("Duplicates appeared in files to load: " + ", ".join(files))

    if cleanup:
        print('LPW L2 cleanup complete')
        return

    if not files:
        raise IOError("No data found")

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)


    if kind == 'lpnt':
        output = dict(time=None, ne=None, te=None, usc=None)
        for f in sorted(files):
            c = cdflib.CDF(f)
            if output['time'] is None:
                # inx =
                output['time'] = c['time_unix']
                output['ne'] = c['data'][:,0]
                output['te'] = c['data'][:,1]
                output['usc'] = c['data'][:,2]
            else:
                output['time'] = np.hstack((output['time'],
                    c['time_unix']))

                for v, i in zip(('ne', 'te', 'usc'), (0,1,2)):
                    output[v] = np.hstack((output[v], c['data'][:,i]))
            c.close()

    elif kind == 'wn':
        output = dict(time=None, ne=None)
        for f in sorted(files):
            print(f)
            c = cdflib.CDF(f)
            if output['time'] is None:
                # inx =
                output['time'] = c['time_unix']
                output['ne'] = c['data']
            else:
                output['time'] = np.hstack((output['time'],
                    c['time_unix']))
                output['ne'] = np.hstack((output['ne'],
                    c['data']))

                # for v, i in zip(('ne', 'te', 'usc'), (0,1,2)):
                #     output[v] = np.hstack((output[v], c.varget('data'][:,i])))
            c.close()


    elif kind == 'wspecact':
        output = dict(time=None, spec=None, freq=None)
        for f in sorted(files):
            print(f)
            c = cdflib.CDF(f)

            if output['time'] is None:
                output['time'] = c['time_unix']
                output['spec'] = c['data'].T
                output['freq'] = c['freq'][0,:]
            else:
                output['time'] = np.hstack((output['time'],
                                c['time_unix']))
                output['spec'] = np.hstack((output['spec'],
                                c['data'].T))
            c.close()

        # print 'Warning: spectra output is not interpolated!'

    elif kind == 'wspecpas':
        output = dict(time=None, spec=None, freq=None)
        for f in sorted(files):
            print(f)
            c = cdflib.CDF(f)

            if output['time'] is None:
                output['time'] = c['time_unix']
                output['spec'] = c['data'].T
                output['freq'] = c['freq'][0,:]
            else:
                output['time'] = np.hstack((output['time'],
                                c['time_unix']))
                output['spec'] = np.hstack((output['spec'],
                                c['data'].T))
        # print 'Warning: spectra output is not interpolated!'
            c.close()

    elif kind == 'lpiv':
        output = dict(time=None, current=None, volt=None)
        for f in sorted(files):
            c = cdflib.CDF(f)

            if output['time'] is None:
                output['time'] = c['time_unix']
                output['current'] = c['data'].T
                output['volt'] = c['volt'].T
            else:
                output['time'] = np.hstack((output['time'],
                                c['time_unix']))
                output['current'] = np.hstack((
                    output['current'], c['data'].T))
                output['volt'] = np.hstack((
                    output['volt'], c['volt'].T))

            c.close()

    else:
        raise ValueError("Input kind='%s' not recognized" % kind)

    output['time'] = output['time'] + celsius.spiceet("1970-01-01T00:00")
    return output

def lpw_plot_spec(s, ax=None, cmap=None, norm=None,
    max_frequencies=512, max_times=2048, fmin=None, fmax=None,
    labels=True, colorbar=True, full_resolution=False):
    """Transform and plot a spectra dictionary generated by lpw_load.
Doesn't interpolate linearly, but just rebins data.  Appropriate for presentation purposes, but don't do science with the results."""

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if cmap is None: cmap = 'Spectral_r'
    if norm is None: norm = LogNorm(1e-16, 1e-8)

    img_obj = plt.pcolormesh(s['time'], s['freq'], s['spec'],
            cmap=cmap, norm=norm)

    plt.yscale('log')
    # plt.xlim(t0, t1)

    if labels:
        plt.ylabel('f / Hz')
    if colorbar:
        cbar = plt.colorbar(cax=celsius.make_colorbar_cax())
        cbar.set_label(r'V$^2$ m$^{-2}$ Hz$^{-1}$')
    else:
        cbar = None

    return img_obj, cbar

def lpw_plot_iv(s, boom=1, ax=None, cmap=None, norm=None,
    start=None, finish=None,
    voltage=None,
    labels=True, colorbar=True, log_abs=True):
    """Plot LP IV sweeps as a time series."""

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if cmap is None:
        plt.set_cmap('viridis')
        if log_abs is False:
            plt.set_cmap('RdBu_r')
        cmap = plt.get_cmap()
        cmap.set_bad('grey')

    if not norm:
        norm = plt.Normalize(1e-7, 1e-7)
        if log_abs:
            norm = LogNorm(1e-9, 1e-5)

    d = s['current']
    if log_abs:
        d = np.abs(d)

    img_obj = plt.pcolormesh(s['time'], s['volt'], d, cmap=cmap, norm=norm)

    if labels:
        plt.ylabel(r'U$_{Bias}$ / V')

    if colorbar:
        cbar = plt.colorbar(cax=celsius.make_colorbar_cax(ax))
        cbar.set_label(r'i / A')
    else:
        cbar = None

    return img_obj, cbar

def cleanup(start=None, finish=None):
    if not start: start = celsius.spiceet("2014-09-22T00:00")
    if not finish: finish = celsius.now()

    # Cleanup commands
    lpw_l2_load(start, finish, cleanup=True, verbose=True)

if __name__ == '__main__':
    import maven
    import mex

    if False:

        t0 = celsius.spiceet("2015-01-07T00:00")
        c = get_hf_act_densities(t0, t0 + 86400.*2., verbose=True)
        print(list(c.keys()))
        inx = c['confidence'] > 95.

        # plt.close('all')
        fig, axs = plt.subplots(3,1, sharex=True)

        plt.sca(axs[0])
        plt.plot(c['time'][inx], c['density'][inx], 'r.')

        # plt.plot(mexdata['time'], mexdata['ne'], 'b.')
        plt.ylabel("ne / cm^-3")

        plt.yscale('log')

        plt.sca(axs[1])
        time = np.linspace(c['time'][0], c['time'][-1], 500)
        plt.plot(time, sun_z_angle(time), 'r-')
        plt.plot(time, ram_z_angle(time), 'k-')
        plt.fill_between(plt.xlim(), (55-10, 55-10), (55+10, 55+10),
                facecolor='b', alpha=0.3, zorder=-99)
        plt.ylabel(r"$\theta$ / deg")
        # plt.ylim(-180., 180.)
        plt.sca(axs[2])
        mso, sza = maven.mso_r_lat_lon_position(time, sza=True)
        plt.plot(time, sza)
        plt.ylabel("SZA / deg")

        celsius.setup_time_axis()
        plt.show()

    if True:
        plt.close('all')
        start = celsius.spiceet("2015-04-23T06:00")
        finish = start + 86400. /2.

        # finish = start + 86400. * 2. - 1.

        xl = np.array((start, finish))
        xo = np.array((1,1))

        o = lpw_l2_load(kind='wspecact', start=start, finish=finish)
        o2 = lpw_l2_load(kind='wn', start=start, finish=finish)
        o3 = lpw_l2_load(kind='lpnt', start=start, finish=finish)

        inx = np.isfinite(o2['ne'])
        ne_w = np.interp(o3['time'], o2['time'][inx], o2['ne'][inx])

        fig, axs = plt.subplots(4,1, sharex=True, figsize=(8,12), gridspec_kw=dict(height_ratios=(5,2,2,2)))

        plt.subplots_adjust(hspace=0.01)
        plt.sca(axs[0])
        lpw_plot_spec(o, colorbar=False, full_resolution=True, fmin=2e4)
        # plt.ylim(1e4, 2e6)
        plt.plot(o2['time'], 8980*np.sqrt(o2['ne']), 'k+')
        plt.plot(o3['time'], 8980*np.sqrt(o3['ne']), 'r+')
        # plt.plot(o3['time'], 8980*np.sqrt(ne_w), 'b*')

        plt.sca(axs[1])
        plt.plot(o3['time'], np.sqrt(o3['ne']/ne_w), 'k.')
        plt.plot(xl, xo * 1., 'b--')
        plt.plot(xl, xo * 2., 'b--')
        plt.plot(xl, xo * np.sqrt(2), 'b:')
        plt.yscale('log')
        plt.ylim(0.1, 10.)
        plt.ylabel(r'$f_{pe,IV} / f_{pe,W}$')

        plt.sca(axs[2])
        plt.plot(o3['time'], o3['te']/11604., 'k.')
        plt.yscale('log')
        plt.ylabel('Te / eV')

        plt.sca(axs[3])
        plt.plot(o3['time'], 0.069 * np.sqrt(o3['te']/11604. / o3['ne']), 'k.')
        plt.plot(o3['time'], 0.069 * np.sqrt(o3['te']/11604. / ne_w), 'r.')

        plt.plot(xl, xo * 0.0063/2., 'b--')
        plt.plot(xl, xo * 0.05/2., 'b:')
        plt.plot(xl, xo * 0.4/2., 'b--')
        plt.ylim(5e-5, 2e-3)
        plt.yscale('log')
        plt.ylabel(r'$\lambda_D$/m')

        celsius.setup_time_axis()

        plt.figure()
        plt.scatter(
            0.069 * np.sqrt(o3['te']/11604. / o3['ne']),
            np.sqrt(o3['ne']/ne_w), c=o3['time'], marker='.', edgecolor='none'
        )
        plt.ylabel(r'$f_{pe,IV} / f_{pe,W}$')
        plt.xlabel(r'$\lambda_D$/m')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(5e-5, 2e-3)
        plt.ylim(0.1, 10.)
        x = np.array((5e-5, 2e-3))
        plt.plot(x, 10.**(-0.24*np.log10(x)-0.7))

        plt.show()
