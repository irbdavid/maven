import celsius
import numpy as np
import os
from glob import glob, iglob

# import spiceypy
import spiceypy

KERNELS_LOADED = False

# NOMINAL_INSERTION_DATE = spiceet("2014 10 27")
INSERTION_DATE = celsius.spiceet('2014-09-22T02:24')
LAUNCH_DATE = celsius.spiceet('2013-11-17T18:28')

# nb: Public SDC access:
# http://lasp.colorado.edu/maven/sdc/public/data/sci


REQUIRED_KERNELS = [
        'lsk/naif*.tls',
        'fk/maven_*.tf',
        'pck/pck*.tpc',
        'sclk/MVN_SCLKSCET.*.tsc',
        'spk/de421.bsp',
        'spk/de430s.bsp',
        'spk/mar097s.bsp',
        'spk/maven_orb.bsp',
        'spk/maven_orb_rec.bsp',
        # 'RSSD0002.TF',
                ]

DIRECTORY = None

# last_spice_time_window = 'NONE_INITIALIZED'

def check_spice_furnsh(*args, **kwargs):
    load_kernels(*args, **kwargs)

def load_kernels(time=None, force=False, verbose=False,
                load_all=False, keep_previous=False):
    """Load spice kernels, with a stateful thing to prevent multiple calls"""
    last_spice_time_window = getattr(spiceypy,
            'last_spice_time_window', 'MVN:NONE')

    if load_all:
        # Launch to now + 10 yrs
        start = celsius.spiceet("2013-11-19T00:00")
        finish = celsius.spiceet(celsius.now() + 10.*86400.*365.)

    if time is None:
        start = None
        finish = None
        start_str = 'NO_TIME_SET'
        finish_str = ''
        start_int=-999999
        finish_int=-999999
    else:
        if hasattr(time, '__len__'):
            start = time[0]
            finish = time[-1]

        else:
            start = time
            finish = time
        start_str = celsius.utcstr(start, 'ISOC')
        finish_str = celsius.utcstr(finish, 'ISOC')
        start_int = int(start_str[2:4] + start_str[5:7] + '01')
        finish_int = int(finish_str[2:4] + finish_str[5:7] + '01')
        start_str = '%06d' % start_int
        finish_str = '%06d' % finish_int

    this_spice_time_window = start_str + finish_str

    if not 'NONE' in last_spice_time_window:
        if last_spice_time_window == this_spice_time_window:
            if verbose:
                print('LOAD_KERNELS [MVN]: Interval unchanged')
            return

        if keep_previous:
            if verbose:
                print('LOAD_KERNELS [MVN]: Keeping loaded kernels')
            return

    spiceypy.last_spice_time_window = 'MVN:' + this_spice_time_window

    spiceypy.kclear()

    try:
        kernel_directory = os.getenv('MAVEN_KERNEL_DIR')
        if verbose:
            print('LOAD_KERNELS [MVN]: Registering kernels:')

        for k in REQUIRED_KERNELS:

            if '*' in k:
                files = glob(kernel_directory + k)
                m = -1
                file_to_load = ''
                for f in files:
                    t = os.path.getmtime(f)
                    if t > m:
                        m = t
                        file_to_load = f
                if verbose:
                    print(file_to_load)
                if file_to_load:
                    spiceypy.furnsh(file_to_load)
                else:
                    raise IOError("No match for %s" % k)

            else:
                spiceypy.furnsh(kernel_directory + k)
                if verbose: print(kernel_directory + k)

        # time sensitive kernels
        load_count = 0

        # used to determine whether or not to load the most recent, unnumbered
        # rolling update kernel
        max_encountered = -99999

        if start_int > -999999:
            # Load time-sensitive kenrels
            for f in iglob(kernel_directory + 'spk/maven_orb_rec_*.bsp'):
                this_start = int(f.split('_')[3])
                this_finish = int(f.split('_')[4])
                if this_finish < start_int: continue
                if this_start > finish_int: continue
                spiceypy.furnsh(f)
                load_count += 1
                if verbose: print(f)

                if this_start > max_encountered: max_encountered = this_start
                if this_finish > max_encountered: max_encountered = this_finish

            if max_encountered < finish_int:
                # load the rolling-update kernel too
                f = kernel_directory + 'spk/maven_orb_rec.bsp'
                # spiceypy.furnsh(f)
                load_count += 1
                if verbose: print(f)

            if load_count == 0:
                raise IOError("No kernels matched for time period")

    except Exception as e:
        spiceypy.kclear()
        spiceypy.last_spice_time_window = 'MVN:NONE_ERROR'
        raise

    print('LOAD_KERNELS [MVN]: Loaded %s' % spiceypy.last_spice_time_window)

def unload_kernels():
    """Unload kernels"""

    # global last_spice_time_window

    try:
        spiceypy.kclear()

        # But, we always want the LSK loaded.  This should be safe provided
        # a) celsius was loaded first (safe assertion, this code won't work
        # without it), meaning that the latest.tls was updated if needs be
        # b) uptime for this instance is less than the lifetime of a tls kernel
        # (years?)
        spiceypy.furnsh(
            os.getenv("SC_DATA_DIR", default=expanduser('~/data/')) + \
            'latest.tls'
        )

        spiceypy.last_spice_time_window = 'MVN:NONE_UNLOADED'
    except RuntimeError as e:
        spiceypy.last_spice_time_window = 'MVN:NONE_ERROR'
        raise e

load_spice_kernels = load_kernels # Synonym, innit
unload_spice_kernels     = unload_kernels

def describe_loaded_kernels(kind='all'):
    """Print a list of loaded spice kernels of :kind:"""

    all_kinds = ('spk', 'pck', 'ck', 'ek', 'text', 'meta')
    if kind == 'all':
        for k in all_kinds:
            describe_loaded_kernels(k)
        return

    n = spiceypy.ktotal(kind)
    if n == 0:
        print('No loaded %s kernels' % kind)
        return

    print("Loaded %s kernels:" % kind)
    for i in range(n):
        data = spiceypy.kdata(i, kind, 100, 10, 100)
        print("\t%d: %s" % (i, data[0]))


def position(time, frame='IAU_MARS', target='MAVEN',
            observer='MARS', correction='NONE', verbose=False):
    """Wrapper around spiceypy.spkpos that handles array inputs, and provides useful defaults"""
    load_kernels(time, verbose=verbose)

    def f(t):
        try:
            pos, lt = spiceypy.spkpos(target, t, frame, correction, observer)
        except spiceypy.support_types.SpiceyError:
            return np.empty(3) + np.nan
        return np.array(pos)

    if hasattr(time, '__iter__'):
        return np.array([f(t) for t in time]).T
    else:
        return f(time)


def iau_mars_position(time, **kwargs):
    """An alias: position(time, frame='IAU_MARS')"""
    return position(time, frame='IAU_MARS', **kwargs)

def mso_position(time, **kwargs):
    """An alias: position(time, frame='MSO')"""
    return position(time, frame='MAVEN_MSO', **kwargs)

def reclat(pos):
    """spiceypy.reclat with a wrapper for ndarrays"""

    check_spice_furnsh(keep_previous=True)

    if isinstance(pos, np.ndarray):
        if len(pos.shape) > 1:
            return np.array([spiceypy.reclat(p) for p in pos.T]).T
    return spiceypy.reclat(pos)

def recpgr(pos, body="MARS"):
    """spiceypy.recpgr for mars, with a wrapper for ndarrays"""
    check_spice_furnsh(keep_previous=True)

    if body == "MARS":
       r, e = 3396.2, 0.005888934691714269
    else:
       raise NotImplemented("Unknown body: " + body)

    def f(p):
        return spiceypy.recpgr(body, p, r, e)
    if isinstance(pos, np.ndarray):
        if len(pos.shape) > 1:
            return np.array([f(p) for p in pos.T]).T
    return f(pos)

# Convert "WEST" longitudes to "EAST", which seems to be more commonly used.
# This is the reason for the non-multiplication of the longitude here, and the -1 in the
# corresponding iau_pgr_alt_lat_lon_position function
def iau_r_lat_lon_position(time, **kwargs):
    """" Return the position of MEX at `time`, in Radial Distance/Latitude/EAST Longitude.
    No accounting for the oblate spheroid is done, hence returning radial distance [km]"""
    tmp = reclat(position(time, frame = 'IAU_MARS', **kwargs))
    out = np.empty_like(tmp)
    out[0] = tmp[0]
    out[1] = np.rad2deg(tmp[2])
    out[2] = np.rad2deg(tmp[1])
    return out

# Convert "WEST" longitudes to "EAST", which seems to be more commonly used.
# This is the reason for the -1 * multiplication of the longitude here, and the *1 in the
# corresponding iau_pgr_alt_lat_lon_position function
def iau_pgr_alt_lat_lon_position(time, **kwargs):
    """ Return the position of MEX at `time`, in Altitude/Latitude/EAST Longitude.
    This accounts also for the oblate-spheriod of Mars"""
    tmp = recpgr(position(time, frame = 'IAU_MARS', **kwargs))
    out = np.empty_like(tmp)
    out[0] = tmp[2]
    out[1] = np.rad2deg(tmp[1])
    out[2] = 360. - 1. * np.rad2deg(tmp[0]) #Convert to EAST LONGITUDE, RECPGR returns in [0, 2pi], so add 360.
    return out

def mso_r_lat_lon_position(time, mso=False, sza=False, **kwargs):
    """Returns position in MSO spherical polar coordinates.
    With `mso' set, return [r/lat/lon], [mso x/y/z [km]].
    With `sza' set, return [r/lat/lon], [sza [deg]].
    With both, return return [r/lat/lon], [mso x/y/z [km]], [sza [deg]]."""

    if sza:
        pos = position(time, frame = 'MAVEN_MSO', **kwargs)
        sza = np.rad2deg(np.arctan2(np.sqrt(pos[1]**2 + pos[2]**2), pos[0]))
        if isinstance(sza, np.ndarray):
            inx = sza < 0.
            if np.any(inx):
                sza[inx] = 180. + sza[inx]
        elif sza < 0.0:
            sza = 180. + sza

        tmp = reclat(pos)
        tmp_out = np.empty_like(tmp)
        tmp_out[0] = tmp[0]
        tmp_out[1] = np.rad2deg(tmp[2])
        tmp_out[2] = np.rad2deg(tmp[1])
        if mso:
            return tmp_out, pos, sza
        return tmp_out, sza

    else:
        pos = position(time, frame = 'MAVEN_MSO', **kwargs)
        tmp = reclat(pos)
        tmp_out = np.empty_like(tmp)
        tmp_out[0] = tmp[0]
        tmp_out[1] = np.rad2deg(tmp[2])
        tmp_out[2] = np.rad2deg(tmp[1])
        if mso:
            return tmp_out, pos
        return tmp_out

def sub_solar_longitude(et):
    """Sub-solar longitude in degrees at `et`."""
    load_kernels(et)
    def f(t):
        pos, lt = spiceypy.spkpos("SUN", t, 'IAU_MARS', "NONE", "MARS")
        return np.array(pos)

    def func(time):
        if hasattr(time, '__iter__'):
            return np.array([f(t) for t in time]).T
        else:
            return f(time)

    tmp = recpgr(func(et))
    return np.rad2deg(tmp[0])

def sub_solar_latitude(et, body='MARS'):
    """Sub-solar latitude in degrees at `et`."""
    load_kernels(et)
    def f(t):
        pos, lt = spiceypy.spkpos("SUN", t, 'IAU_' + body, "NONE", body)
        return np.array(pos)

    def func(time):
        if hasattr(time, '__iter__'):
            return np.array([f(t) for t in time]).T
        else:
            return f(time)

    tmp = recpgr(func(et))
    return np.rad2deg(tmp[1])

def modpos(x, radians=False, min=0.):
    if radians:
        return (x % (2. * np.pi) + 2. * np.pi) % (2. * np.pi)
    return (x % (360.) + 360.) % 360.


def read_maven_orbits(fname):
    """docstring for read_maven_orbits"""

    print('Reading %s ... ' % fname, end=' ')

    orbit_list = celsius.OrbitDict()

    f = open(fname, 'r')

    # skip headers
    f.readline()
    f.readline()

    # Lockheed putting in the periapsis of the orbit, and the terminating apoapsis, i.e. wrong way round
    last_apoapsis = np.nan

    for line in f.readlines():
        if "Unable to determine" in line:
            print("Detected 'Unable to determine' (last orbit bounds error?)")
            continue

        try:
            number = int(line[0:5])
            apoapsis = celsius.spiceet(line[51:71])
            periapsis = celsius.spiceet(line[7:27])

            if ~np.isfinite(last_apoapsis):
                this_apo = periapsis - 0.00000001 # A hack.  Don't want a NaN in there.
                # First orbit will start at periapsis, effectively
            else:
                this_apo = last_apoapsis
            last_apoapsis = apoapsis
            m = celsius.Orbit(number=number, start=this_apo,
                    periapsis=periapsis, apoapsis=apoapsis, name='MAVEN')

            orbit_list[number] = m
        except ValueError as e:
            print(e)
            raise

    print(' read %d orbits (MAX = %d)' % (len(orbit_list), number))
    return orbit_list

if __name__ == '__main__':
    import pylab as plt
    load_kernels()
    describe_loaded_kernels()
    plt.close('all')

    if True:
        plt.figure()

        t0 = celsius.spiceet("2013-001T00:00")
        t1 = celsius.spiceet("2016-001T00:00")
        for t in np.arange(t0, t1, 86400.):
            try:
                pos = position(t)
                plt.plot(t, np.sqrt(pos[0]**2. + pos[1]**2. + pos[2]**2.),
                        'ko')
            except Exception as e:
                print(t, e)
        plt.yscale('log')
        setup_time_axis()
        plt.show()
