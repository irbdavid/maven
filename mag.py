import numpy as np
import pylab as plt
import spiceypy
import celsius
from . import sdc_interface
from functools import wraps

import os

def load_mag_l2(start, finish, kind='ss1s',
        http_manager=None, delete_others=True, cleanup=False, verbose=None):
    kind = kind.lower()

    if not delete_others:
        raise RuntimeError("Not written yet")

    if http_manager is None:
        http_manager = sdc_interface.maven_http_manager

    t = celsius.CelsiusTime(celsius.utcstr(start)[0:8]+"T00:00")

    #  Each month:
    # /maven/data/sci/mag/l2/2017/12/mvn_mag_l2_2017365ss1s_20171231_v01_r01.sts
    files = []
    while t < finish:
        files.extend(
                http_manager.query(
#        'mag/l2/%04d/%02d/mvn_mag_l2_%04d%03d%s_%04d%02d%02d_v*_r*.sts' % \
#                                            (t.year, t.month, t.year, t.doy, kind, t.year,t.month,t.day),
        'mag/l2/%04d/%02d/mvn_mag_l2_%04d%03d%s_*_v*_r*.sts' %
                    (t.year, t.month, t.year, t.doy, kind),
                    start=start, finish=finish,
                    version_function=\
                        lambda x: (x[0], float(x[1]) + float(x[2])/100.),
                    date_function=lambda x: sdc_interface.yyyymmdd_to_spiceet(x[0]),
                    cleanup=cleanup, verbose=verbose
                )
            )
        t = celsius.CelsiusTime(t.spiceet + 86400.)

    if cleanup:
        print('MAG L2 Cleanup complete')
        return

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)


    if kind == 'ss1s':
        output = {'time':None, 'def':None}
        for f in sorted(files):

            c = np.loadtxt(f, skiprows=145, usecols=[6,7,8,9]).T
            c[0] = c[0] * 86400. + celsius.spiceet(
                f.split('_')[-3][:4] + '-001T00:00'
            )
            if output['time'] is None:
                output['time'] = np.array(c[0])
                output['b'] = np.array(c[1:])

            else:
                output['time'] = np.hstack((output['time'],np.array(c[0])))
                output['b'] = np.hstack((output['b'],np.array(c[1:])))

    else:
        raise ValueError("Input kind='%s' not recognized" % kind)

    output['time'] = output['time'] + celsius.spiceet("1970-01-01T00:00")

    return output

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    start = celsius.spiceet('2016-09-01T00:00')

    data = load_mag_l2(start, start+86400.-1.)
    print(data['time'].shape)
    print(data['b'].shape)

    plt.plot(data['time'], data['b'][0])
    plt.plot(data['time'], data['b'][1])
    plt.plot(data['time'], data['b'][2])
    plt.show()
