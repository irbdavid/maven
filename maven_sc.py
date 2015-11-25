import celsius
import numpy as np
import os
from glob import glob

# import spiceypy
import spiceypy

KERNELS_LOADED = False

# NOMINAL_INSERTION_DATE = spiceet("2014 10 27")
INSERTION_DATE = celsius.spiceet('2014-09-22T02:24')
LAUNCH_DATE = celsius.spiceet('2013-11-17T18:28')

# nb: Public SDC access:
# http://lasp.colorado.edu/maven/sdc/public/data/sci


REQUIRED_KERNELS = [
        '../spg/data/misc/spice/naif/MAVEN/kernels/fk/maven_*.tf',
        '../spg/data/misc/spice/naif/MAVEN/kernels/lsk/naif*.tls',
        '../spg/data/misc/spice/naif/MAVEN/kernels/pck/pck*.tpc',
        '../spg/data/misc/spice/naif/MAVEN/kernels/sclk/MVN_SCLKSCET.*.tsc',
        '../spg/data/misc/spice/naif/MAVEN/kernels/spk/de421.bsp',
        '../spg/data/misc/spice/naif/MAVEN/kernels/spk/de430s.bsp',
        '../spg/data/misc/spice/naif/MAVEN/kernels/spk/mar097s.bsp',
        '../spg/data/misc/spice/naif/MAVEN/kernels/spk/maven_orb.bsp',
        '../spg/data/misc/spice/naif/MAVEN/kernels/spk/maven_orb_rec_*.bsp',
        '../spg/data/misc/spice/naif/MAVEN/kernels/spk/maven_orb_rec.bsp',
        # 'RSSD0002.TF',
                ]

DIRECTORY = None
MAVEN_DATA_DIRECTORIES = ['/Volumes/ETC/data/maven/spice/']

def load_kernels(force=False, directory=DIRECTORY):
    global KERNELS_LOADED

    if not force:
        if KERNELS_LOADED: return

    if not directory:
        # directory = os.getenv("SC_DATA_DIR") + 'maven/spg/data/misc/spice/naif/MAVEN/kernels/'
        directory = os.getenv("SC_DATA_DIR") + 'maven/kernels/'

    try:
        for k in REQUIRED_KERNELS:
            fname = directory + k
            if '*' in fname:
                matches = glob(fname)
                if not matches:
                    raise IOError("Missing: %s" % fname)
                matches = sorted(matches, key=lambda x:os.path.getmtime(x))
                for m in matches:
                    spiceypy.furnsh(m)
            else:
                spiceypy.furnsh(fname)

    except Exception as e:
        KERNELS_LOADED = False
        for k in REQUIRED_KERNELS:
            try:
                spiceypy.unload(directory + k)
            except IOError as e:
                pass
        raise e

    KERNELS_LOADED = True

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
            observer='MARS', correction='NONE'):
    """Wrapper around spiceypy.spkpos that handles array inputs, and provides useful defaults"""
    load_kernels()
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


def iau_mars_position(time):
    """An alias: position(time, frame='IAU_MARS')"""
    return position(time, frame='IAU_MARS')

def mso_position(time):
    """An alias: position(time, frame='MSO')"""
    return position(time, frame='MAVEN_MSO')

def reclat(pos):
    """spiceypy.reclat with a wrapper for ndarrays"""
    if isinstance(pos, np.ndarray):
        if len(pos.shape) > 1:
            return np.array([spiceypy.reclat(p) for p in pos.T]).T
    return spiceypy.reclat(pos)

def recpgr(pos, body="MARS"):
    """spiceypy.recpgr for mars, with a wrapper for ndarrays"""

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
    load_kernels()
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
    load_kernels()
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
