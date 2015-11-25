=============================================================
Python library for accessing and plotting MAVEN Level 2 data.
=============================================================

Introduction
------------
This module provides an interface to the MAVEN Science Data Center (SDC), hosted at LASP.
    `https://lasp.colorado.edu/maven/sdc/public`
Routines are provided that can query the SDC for the latest MAVEN data, download to a local mirror, delete obsolete data, and read and plot said data using matplotlib.  MAVEN science team members with access to the private area of the SDC can also use these routines by supplying their credentials in a shell variable.

The set of instruments and data products supported for reading and plotting is evolving, and contributions to this are very welcome.

Requirements
------------

1. Tested against the anaconda python distrubition (v 3.5)
2. SpiceyPy library required for NAIF spice interface
3. SpacePy library required for CDF access
4. celsius library

Installation
------------

1. Satisfy requirements above
2. Add this module into your python path.
3. If you have team-level SDC access, a shell variable needs to be set
containing your username and password:
    `export MAVENPFP_USER_PASS=username:password`
4. Set the local directory that will be used as a local mirror of the SDC:
    `export SC_DATA_DIR="~/data"`
5. Run tests?


Examples of use
---------------
```python
import maven
import celsius
import matplotlib.pyplot as plt
import numpy as np

start = celsius.spiceet("2015-05-01")
finish = start + 86400. -1. # just to avoid loading two files instead of one.

lpw_data = maven.lpw.lpw_l2_load(kind='lpnt', start=start, finish=finish)

fig, axs = plt.subplots(3,1,sharex=True)

plt.sca(axs[0])
plt.plot(lpw_data['time'], lpw_data['ne'], 'k.')

plt.sca(axs[1])
swea_data = maven.swea.load_swea_l2_summary(start, finish)
maven.swea.plot_swea_l2_summary(swea_data)

time = np.linspace(start, finish, 128)
pos_mso = maven.mso_position(time)

plt.sca(axs[2])
plt.plot(time, pos_mso[0], 'r-')
plt.plot(time, pos_mso[1], 'g-')
plt.plot(time, pos_mso[2], 'b-')

celsius.setup_time_axis()

plt.show()
```
