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

1. Tested against the anaconda python distrubition (v 3.5 `https://www.continuum.io/downloads`)
2. SpiceyPy library required for NAIF spice interface `https://github.com/AndrewAnnex/SpiceyPy`
3. SpacePy library required for CDF access `http://spacepy.lanl.gov/`
4. celsius library `https://github.com/irbdavid/celsius`

Installation
------------

1. Satisfy requirements above
2. Add this module into your python path.
3. If you have team-level SDC access, a shell variable needs to be set
containing your username and password:
    `export MAVENPFP_USER_PASS=username:password`
4. At a minimum, set the local directory that will be used for data storage:
    `export SC_DATA_DIR="~/data"`
4. To provide comapatability with a parallel use of the Berkeley IDL code base, `MAVEN_DATA_DIR` and `MAVEN_KERNEL_DIR` can be used to specify the locations of existing local data.
5. Run tests?


Known Issues
------------

* This software is un-tested, and will definitely contain bugs.  Please report them, or patch them, but don't ignore them.  
* MAVEN L2 data is still evolving, and support is not included here for reading 'old' versions of files.  Only the most recent version and release of any file is obtained, and this has occasionally lead to broken compatibility, e.g. if new columns are added or re-ordered in an ASCII file.  CDF files are more robust in this respect.
* Reading of KP data is implemented, but the naming of variables will not match the IDL toolkit, as the necessary information is not contained in the KP files themselves.  Instead, shortened variable names are generated on the fly using a small lookup table, e.g.:
    'H+ temperature STATIC' becomes 'static.hplus_temp'
While not ideal, this should at least be 'safe', and prevent naming accidents that could easily occur with something hard-coded.


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
