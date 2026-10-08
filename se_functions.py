import numpy as np
from astropy.table import Table


def find_bursts(data, thresh=1.0, gap_myr=0.1, min_dur_myr=0.001):
    t = np.array(data['time']) * 1e3
    f = np.array(data['f_edd'])
    dotMaccr = np.array(data['dotMaccr'])
    dotMedd = np.array(data['dotMedd'])
    ratio_edd = dotMaccr / dotMedd

    m = ratio_edd > 0
    ratio_edd[m]

    above = f > thresh

    d = np.diff(above.astype(int))
    starts = np.flatnonzero(d == 1) + 1
    ends   = np.flatnonzero(d == -1) + 1

    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends,len(f)]

    merged = []
    for s, e in zip(starts, ends):
        if merged and t[s] - t[merged[-1][1] - 1] < gap_myr:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    rows = []
    for s, e in merged:
        dur = t[e - 1] - t[s]
        if dur < min_dur_myr:
            continue
        rows.append(dict(
            i_start=s, i_end=e - 1,
            t_start_myr=t[s], t_end_myr=t[e - 1], duration_myr=dur,
            z_start=data['z'][s], z_end=data['z'][e-1],
            aexp_start=float(data['aexp'][s]), aexp_end=float(data['aexp'][e - 1]),
            M_start=float(data['BHmass'][s]),
            M_end=float(data['BHmass'][e]),
            dM_accr=float(np.sum(np.asarray(data['dMgrowth'])[s:e])),
            E_feed=np.sum(data['Efeed'][s:e]),
            f_edd_max=float(f[s:e].max()),
            dM_edd_max=float(ratio_edd[s:e].max())
        ))
    return Table(rows=rows)

def hello():
    print("hello")