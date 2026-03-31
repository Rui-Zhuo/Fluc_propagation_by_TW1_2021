import numpy as np
import spiceypy as spice
import pandas as pd
from matplotlib import pyplot as plt

spice.furnsh('./kernels/de430.bsp')
spice.furnsh('./kernels/naif0012.tls')
spice.furnsh('./kernels/pck00010.tpc')
spice.furnsh('./kernels/earth_000101_240326_240101.bpc')
spice.furnsh('./kernels/earth_000101_240326_240101.cmt')
spice.furnsh('./kernels/mars_iau2000_v1.tpc')
Rs_km = 696000
AU_km = 1.5e8


def create_epoch(range_dt, step_td):
    beg_dt = range_dt[0]
    end_dt = range_dt[1]
    return [beg_dt + n * step_td for n in range((end_dt - beg_dt) // step_td)]


def get_body_pos(bodyName, epochDt, coord='IAU_SUN'):
    epochEt = spice.datetime2et(epochDt)
    bodyPos, _ = spice.spkpos(bodyName, epochEt, coord, 'NONE', 'SUN')
    return bodyPos


def get_station_pos(stationName, epochDt, coord='IAU_SUN'):
    epochEt = spice.datetime2et(epochDt)
    df = pd.read_csv('coordinate_list.txt', sep='\s+', header=None, names=['StationName', 'Number', 'x', 'y', 'z'])
    dfStations = df[df['StationName'] == stationName]
    stationPosItrs = np.array([dfStations.x.values, dfStations.y.values, dfStations.z.values]).squeeze() / 1e3
    stationPosHelioCentric, _ = spice.spkcpt(stationPosItrs, 'EARTH', 'ITRF93', epochEt, coord, 'OBSERVER', 'LT', 'SUN')
    return stationPosHelioCentric[:3]


def plot_SME(startDt, endDt, stepDt, POS_type='EM'):
    epochDt = create_epoch([startDt, endDt], stepDt)
    earthPos = np.array(get_body_pos('EARTH', epochDt, ))
    marsPos = np.array(get_body_pos('MARS BARYCENTER', epochDt))
    # stationPos = get_station_pos('SH',epochDt,)
    if POS_type == 'EM':
        vecPOSn = np.array((earthPos-marsPos) / np.linalg.norm((earthPos-marsPos).T))
    elif POS_type == 'ES':
        vecPOSn = np.array(earthPos / np.linalg.norm(earthPos.T))
    projPos = np.zeros_like(earthPos)
    for i in range(len(epochDt)):
        OE = earthPos[i]
        OM = marsPos[i]
        Nvec = vecPOSn[i]
        OP = (np.dot(OE, Nvec) * OM - np.dot(OM, Nvec) * OE) / (np.dot(OE, Nvec) - np.dot(OM, Nvec))
        projPos[i] = OP
    # %%
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(0, 0, s=1, c='r', label='Sun')
    plt.gca().add_patch(plt.Circle((0, 0), 696300., linewidth=1, color='r'))
    plt.plot(earthPos[:, 0], earthPos[:, 1], c='b', label='Earth')
    plt.plot(marsPos[:, 0], marsPos[:, 1], c='orange', label='Mars')
    plt.plot(projPos[:, 0], projPos[:, 1], c='k', label='Projections')
    for i in range(len(epochDt)):
        plt.plot([earthPos[i, 0], marsPos[i, 0]], [earthPos[i, 1], marsPos[i, 1]], linewidth=1., c='gray')
        plt.plot([0, projPos[i, 0]], [0, projPos[i, 1]], linewidth=1., c='pink')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    # plt.xlim([-1.5e8,2.e8])
    # plt.ylim([-1.5e8,2.5e8])
    plt.legend()
    plt.gca().set_aspect(1)

    plt.subplot(1, 2, 2)
    plt.scatter(0, 0, s=1, c='r', label='Sun')
    plt.gca().add_patch(plt.Circle((0, 0), 696000. / Rs_km, linewidth=1, color='r'))
    plt.plot(earthPos[:, 0] / Rs_km, earthPos[:, 1] / Rs_km, c='b', label='Earth')
    plt.plot(marsPos[:, 0] / Rs_km, marsPos[:, 1] / Rs_km, c='orange', label='Mars')
    plt.plot(projPos[:, 0] / Rs_km, projPos[:, 1] / Rs_km, c='k', label='Projections')
    for i in range(len(epochDt)):
        plt.plot([earthPos[i, 0] / Rs_km, marsPos[i, 0] / Rs_km], [earthPos[i, 1] / Rs_km, marsPos[i, 1] / Rs_km],
                 linewidth=1., c='gray')
        plt.plot([0, projPos[i, 0] / Rs_km], [0, projPos[i, 1] / Rs_km], linewidth=1., c='pink')
    plt.xlabel('X (Rs)')
    plt.ylabel('Y (Rs)')
    plt.xlim([-6, 4])
    plt.ylim([-3, 7])
    # plt.legend()
    plt.gca().set_aspect(1)
    plt.suptitle(startDt.strftime('%Y/%m/%d %H:%M') + ' - ' + endDt.strftime('%Y/%m/%d %H:%M'))
    plt.show()
    return projPos


if __name__ == '__main__':
    from datetime import datetime, timedelta

    startDt = datetime(2021, 10, 4)
    endDt = datetime(2021, 10, 5)
    stepDt = timedelta(hours=6)
    station1Name = 'SH'
    station2Name = 'BJ'
    OP_list = plot_SME(startDt,endDt,stepDt)
    # %%
    epochDt = create_epoch([startDt, endDt], stepDt)
    epochDt = startDt
    earthPos = np.array(get_body_pos('EARTH', epochDt, ))
    marsPos = np.array(get_body_pos('MARS BARYCENTER', epochDt))

    vecPOSn = np.array(earthPos / np.linalg.norm(earthPos.T))
    vecPOSx = np.cross([0, 0, 1], vecPOSn)
    vecPOSx = vecPOSx / np.linalg.norm(vecPOSx)
    vecPOSy = np.cross(vecPOSn, vecPOSx)
    vecPOSy = vecPOSy / np.linalg.norm(vecPOSy)

    station1Pos = get_station_pos(station1Name, epochDt)
    station2Pos = get_station_pos(station2Name, epochDt)
    OM = marsPos
    Nvec = vecPOSn
    proj1Pos = (np.dot(station1Pos, Nvec) * OM - np.dot(OM, Nvec) * station1Pos) / (
                np.dot(station1Pos, Nvec) - np.dot(OM, Nvec))
    proj2Pos = (np.dot(station2Pos, Nvec) * OM - np.dot(OM, Nvec) * station2Pos) / (
                np.dot(station2Pos, Nvec) - np.dot(OM, Nvec))

    proj1Pos_xPOS, proj1Pos_yPOS = np.dot(proj1Pos, vecPOSx), np.dot(proj1Pos, vecPOSy)
    proj2Pos_xPOS, proj2Pos_yPOS = np.dot(proj2Pos, vecPOSx), np.dot(proj2Pos, vecPOSy)

    # %%
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.scatter(0, 0, s=1, c='r', label='Sun')
    plt.gca().add_patch(plt.Circle((0, 0), 696300. / AU_km, linewidth=1, color='r'))
    plt.scatter(earthPos[0] / AU_km, earthPos[1] / AU_km, s=10, c='b', label='Earth')
    plt.scatter(marsPos[0] / AU_km, marsPos[1] / AU_km, s=10, c='orange', label='Mars')
    plt.scatter(proj1Pos[0] / AU_km, proj1Pos[1] / AU_km, s=10, c='green', label=station1Name + ' Projection')
    plt.scatter(proj2Pos[0] / AU_km, proj2Pos[2] / AU_km, s=10, c='magenta', label=station2Name + ' Projection')

    plt.plot([earthPos[0] / AU_km, marsPos[0] / AU_km], [earthPos[1] / AU_km, marsPos[1] / AU_km], linewidth=1.,
             c='gray')
    plt.plot([0, proj1Pos[0] / AU_km], [0, proj1Pos[1] / AU_km], linewidth=1., c='pink')
    plt.plot([0, proj2Pos[0] / AU_km], [0, proj2Pos[1] / AU_km], linewidth=1., c='pink')
    plt.xlabel('X (AU)')
    plt.ylabel('Y (AU)')
    plt.legend()
    plt.gca().set_aspect(1)
    plt.title('Full View (AU scale)')

    plt.subplot(1, 3, 2)
    plt.scatter(0, 0, s=1, c='r', label='Sun')
    plt.gca().add_patch(plt.Circle((0, 0), 696000. / Rs_km, linewidth=1, color='r'))
    plt.scatter(earthPos[0] / Rs_km, earthPos[1] / Rs_km, c='b', label='Earth')
    plt.scatter(marsPos[0] / Rs_km, marsPos[1] / Rs_km, s=10,c='orange', label='Mars')
    plt.scatter(proj1Pos[0] / Rs_km, proj1Pos[1] / Rs_km,s=10, c='green', label=station1Name + ' Projection')
    plt.scatter(proj2Pos[0] / Rs_km, proj2Pos[1] / Rs_km,s=10,c='magenta', label=station2Name + ' Projection')

    plt.plot([station1Pos[0] / Rs_km, marsPos[0] / Rs_km], [station1Pos[1] / Rs_km, marsPos[1] / Rs_km], linewidth=1.,
             c='gray')
    plt.plot([station2Pos[0] / Rs_km, marsPos[0] / Rs_km], [station2Pos[1] / Rs_km, marsPos[1] / Rs_km],
             linewidth=1., c='lightblue')

    plt.plot([0, proj1Pos[0] / Rs_km], [0, proj1Pos[1] / Rs_km], linewidth=1., c='pink')
    plt.plot([0, proj2Pos[0] / Rs_km], [0, proj2Pos[1] / Rs_km], linewidth=1., c='pink')
    plt.xlabel('X (Rs)')
    plt.ylabel('Y (Rs)')
    plt.xlim([-6, 4])
    plt.ylim([-3, 7])
    plt.gca().set_aspect(1)
    plt.title('Near Sun View (Rs scale)')

    plt.subplot(1, 3, 3)
    plt.scatter(proj1Pos_xPOS, proj1Pos_yPOS, s=10, c='green', label=station1Name + ' Projection')
    plt.scatter(proj2Pos_xPOS, proj2Pos_yPOS, s=10, c='magenta', label=station2Name + ' Projection')
    plt.plot([proj1Pos_xPOS, proj2Pos_xPOS], [proj1Pos_yPOS, proj2Pos_yPOS], 'k--')
    plt.plot([0, proj1Pos_xPOS], [0, proj1Pos_yPOS], linewidth=1, c='lightgreen')
    plt.plot([0, proj2Pos_xPOS], [0, proj2Pos_yPOS], linewidth=1, c='pink')
    plt.title('Plane of Sky')
    plt.xlim([proj1Pos_xPOS - 1000, proj1Pos_xPOS + 1000])
    plt.ylim([proj1Pos_yPOS - 1000, proj1Pos_yPOS + 1000])
    plt.gca().set_aspect(1)
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.suptitle('2021-10-04')
    plt.show()
