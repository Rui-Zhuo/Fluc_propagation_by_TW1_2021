import numpy as np
import spiceypy as spice
import pandas as pd
from matplotlib import pyplot as plt

spice.furnsh('E:/Research/Program/else/Sun_Mars_Earth/kernels/de430.bsp')
spice.furnsh('E:/Research/Program/else/Sun_Mars_Earth/kernels/naif0012.tls')
spice.furnsh('E:/Research/Program/else/Sun_Mars_Earth/kernels/pck00010.tpc')
spice.furnsh('E:/Research/Program/else/Sun_Mars_Earth/kernels/earth_000101_240326_240101.bpc')
spice.furnsh('E:/Research/Program/else/Sun_Mars_Earth/kernels/earth_000101_240326_240101.cmt')
spice.furnsh('E:/Research/Program/else/Sun_Mars_Earth/kernels/mars_iau2000_v1.tpc')
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
    df = pd.read_csv('E:/Research/Program/else/Sun_Mars_Earth/coordinate_list.txt', sep='\s+', header=None, names=['StationName', 'Number', 'x', 'y', 'z'])
    dfStations = df[df['StationName'] == stationName]
    stationPosItrs = np.array([dfStations.x.values, dfStations.y.values, dfStations.z.values]).squeeze() / 1e3
    stationPosHelioCentric, _ = spice.spkcpt(stationPosItrs, 'EARTH BARYCENTER', 'ITRF93', epochEt, coord, 'OBSERVER', 'LT', 'SUN')
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
    plt.gca().add_patch(plt.Circle((0, 0), 696300./AU_km, linewidth=1, color='r'))
    plt.plot(earthPos[:, 0]/AU_km, earthPos[:, 1]/AU_km, c='b', label='Earth')
    plt.plot(marsPos[:, 0]/AU_km, marsPos[:, 1]/AU_km, c='orange', label='Mars')
    plt.plot(projPos[:, 0]/AU_km, projPos[:, 1]/AU_km, c='k', label='Projections')
    for i in range(len(epochDt)):
        plt.plot([earthPos[i, 0]/AU_km, marsPos[i, 0]/AU_km], [earthPos[i, 1]/AU_km, marsPos[i, 1]/AU_km], linewidth=1., c='gray')
        plt.plot([0, projPos[i, 0]/AU_km], [0, projPos[i, 1]/AU_km], linewidth=1., c='pink')
    plt.xlabel('X (AU)')
    plt.ylabel('Y (AU)')
    # plt.xlim([-1.5e8,2.e8])
    # plt.ylim([-1.5e8,2.5e8])
    plt.legend()
    plt.title('Full View')
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
    plt.title('Near Sun View')
    plt.suptitle(startDt.strftime('%Y/%m/%d %H:%M') + ' - ' + endDt.strftime('%Y/%m/%d %H:%M'))
    plt.show()
    return projPos

def calculate_project_distance(proj1Pos_xPOS, proj1Pos_yPOS, proj2Pos_xPOS, proj2Pos_yPOS):
    delta_x = proj1Pos_xPOS - proj2Pos_xPOS
    delta_y = proj1Pos_yPOS - proj2Pos_yPOS
    distance = np.sqrt(np.square(delta_x) + np.square(delta_y))
    return distance

def calculate_angle_wrt_radial(proj1Pos_xPOS, proj1Pos_yPOS, proj2Pos_xPOS, proj2Pos_yPOS):
    vecRadial = np.array([(proj1Pos_xPOS + proj2Pos_xPOS) / 2, (proj1Pos_yPOS + proj2Pos_yPOS) / 2])
    vecConnect = np.array([proj2Pos_xPOS - proj1Pos_xPOS, proj2Pos_yPOS - proj1Pos_yPOS])
    vecRadial_magni = np.linalg.norm(vecRadial)
    vecConnect_magni = np.linalg.norm(vecConnect)
    
    angle = np.arccos(np.dot(vecRadial, vecConnect) / (vecRadial_magni * vecConnect_magni))
    return np.degrees(angle), vecRadial_magni / Rs_km


if __name__ == '__main__':
    from datetime import datetime, timedelta

    epochDt = datetime(2021, 10, 4, 8, 0, 0)
    df = pd.read_csv('E:/Research/Program/else/Sun_Mars_Earth/coordinate_list.txt', sep='\s+', header=None, names=['StationName', 'Number', 'x', 'y', 'z'])
    stationNameList = ['JS', 'BD', 'BD', 'BD', 'BD']#df['StationName'].values
    station1Name = stationNameList[0]
    station2Name = stationNameList[1]
    station3Name = stationNameList[2]
    station4Name = stationNameList[3]
    station5Name = stationNameList[4]

    # %%
    earthPos = np.array(get_body_pos('EARTH', epochDt, ))
    marsPos = np.array(get_body_pos('MARS BARYCENTER', epochDt))

    vecPOSn = np.array(earthPos / np.linalg.norm(earthPos.T))
    vecPOSx = np.cross([0, 0, 1], vecPOSn)
    vecPOSx = vecPOSx / np.linalg.norm(vecPOSx)
    vecPOSy = np.cross(vecPOSn, vecPOSx)
    vecPOSy = vecPOSy / np.linalg.norm(vecPOSy)

    OM = marsPos
    Nvec = vecPOSn

    station1Pos = get_station_pos(station1Name, epochDt)
    station2Pos = get_station_pos(station2Name, epochDt)
    station3Pos = get_station_pos(station3Name, epochDt)
    station4Pos = get_station_pos(station4Name, epochDt)
    station5Pos = get_station_pos(station5Name, epochDt)

    proj1Pos = (np.dot(station1Pos, Nvec) * OM - np.dot(OM, Nvec) * station1Pos) / (
                np.dot(station1Pos, Nvec) - np.dot(OM, Nvec))
    proj2Pos = (np.dot(station2Pos, Nvec) * OM - np.dot(OM, Nvec) * station2Pos) / (
                np.dot(station2Pos, Nvec) - np.dot(OM, Nvec))
    proj3Pos = (np.dot(station3Pos, Nvec) * OM - np.dot(OM, Nvec) * station3Pos) / (
                np.dot(station3Pos, Nvec) - np.dot(OM, Nvec))
    proj4Pos = (np.dot(station4Pos, Nvec) * OM - np.dot(OM, Nvec) * station4Pos) / (
                np.dot(station4Pos, Nvec) - np.dot(OM, Nvec))
    proj5Pos = (np.dot(station5Pos, Nvec) * OM - np.dot(OM, Nvec) * station5Pos) / (
                np.dot(station5Pos, Nvec) - np.dot(OM, Nvec))
    
    projPOSx_avg = np.mean([proj1Pos[0], proj2Pos[0], proj3Pos[0]])
    projPOSy_avg = np.mean([proj1Pos[1], proj2Pos[1], proj3Pos[1]])
    projPOSz_avg = np.mean([proj1Pos[2], proj2Pos[2], proj3Pos[2]])
    print('mean postion: ', projPOSx_avg, projPOSy_avg, projPOSz_avg)

    proj1Pos_xPOS, proj1Pos_yPOS = np.dot(proj1Pos, vecPOSx), np.dot(proj1Pos, vecPOSy)
    proj2Pos_xPOS, proj2Pos_yPOS = np.dot(proj2Pos, vecPOSx), np.dot(proj2Pos, vecPOSy)
    proj3Pos_xPOS, proj3Pos_yPOS = np.dot(proj3Pos, vecPOSx), np.dot(proj3Pos, vecPOSy)
    proj4Pos_xPOS, proj4Pos_yPOS = np.dot(proj4Pos, vecPOSx), np.dot(proj4Pos, vecPOSy)
    proj5Pos_xPOS, proj5Pos_yPOS = np.dot(proj5Pos, vecPOSx), np.dot(proj5Pos, vecPOSy)
    
    ## output projected location
    print(station1Name+'(x,y): ', np.round(proj1Pos_xPOS,3), np.round(proj1Pos_yPOS,3))
    print(station2Name+'(x,y): ', np.round(proj2Pos_xPOS,3), np.round(proj2Pos_yPOS,3))
    print(station3Name+'(x,y): ', np.round(proj3Pos_xPOS,3), np.round(proj3Pos_yPOS,3))
    print(station4Name+'(x,y): ', np.round(proj4Pos_xPOS,3), np.round(proj4Pos_yPOS,3))
    print(station5Name+'(x,y): ', np.round(proj5Pos_xPOS,3), np.round(proj5Pos_yPOS,3))

    ## plot figures
    plt.figure(figsize=(6,6))
    ax=plt.subplot(1, 1, 1)
           
    plt.plot([0, proj1Pos_xPOS / Rs_km], [0, proj1Pos_yPOS / Rs_km], linewidth=2, c='red') # 'orange'
    plt.plot([0, proj2Pos_xPOS / Rs_km], [0, proj2Pos_yPOS / Rs_km], linewidth=2, c='blue') # 'green'
    # plt.plot([0, proj3Pos_xPOS / Rs_km], [0, proj3Pos_yPOS / Rs_km], linewidth=2, c='cyan')
    # plt.plot([0, proj4Pos_xPOS / Rs_km], [0, proj4Pos_yPOS / Rs_km], linewidth=2, c='magenta')
    # plt.plot([0, proj5Pos_xPOS / Rs_km], [0, proj5Pos_yPOS / Rs_km], linewidth=2, c='purple')
    
    plt.plot([proj1Pos_xPOS / Rs_km, proj2Pos_xPOS / Rs_km], [proj1Pos_yPOS / Rs_km, proj2Pos_yPOS / Rs_km], 'k--', linewidth=2)
    plt.plot([proj1Pos_xPOS / Rs_km, proj3Pos_xPOS / Rs_km], [proj1Pos_yPOS / Rs_km, proj3Pos_yPOS / Rs_km], 'k--', linewidth=2)
    # plt.plot([proj2Pos_xPOS / Rs_km, proj3Pos_xPOS / Rs_km], [proj2Pos_yPOS / Rs_km, proj3Pos_yPOS / Rs_km], 'k--', linewidth=2)
    # plt.plot([proj1Pos_xPOS / Rs_km, proj4Pos_xPOS / Rs_km], [proj1Pos_yPOS / Rs_km, proj4Pos_yPOS / Rs_km], 'k--', linewidth=2)
    # plt.plot([proj2Pos_xPOS / Rs_km, proj4Pos_xPOS / Rs_km], [proj2Pos_yPOS / Rs_km, proj4Pos_yPOS / Rs_km], 'k--', linewidth=2)
    # plt.plot([proj3Pos_xPOS / Rs_km, proj4Pos_xPOS / Rs_km], [proj3Pos_yPOS / Rs_km, proj4Pos_yPOS / Rs_km], 'k--', linewidth=2)
    # plt.plot([proj1Pos_xPOS / Rs_km, proj5Pos_xPOS / Rs_km], [proj1Pos_yPOS / Rs_km, proj5Pos_yPOS / Rs_km], 'k--', linewidth=2)
    # plt.plot([proj2Pos_xPOS / Rs_km, proj5Pos_xPOS / Rs_km], [proj2Pos_yPOS / Rs_km, proj5Pos_yPOS / Rs_km], 'k--', linewidth=2)
    # plt.plot([proj3Pos_xPOS / Rs_km, proj5Pos_xPOS / Rs_km], [proj3Pos_yPOS / Rs_km, proj5Pos_yPOS / Rs_km], 'k--', linewidth=2)
    # plt.plot([proj4Pos_xPOS / Rs_km, proj5Pos_xPOS / Rs_km], [proj4Pos_yPOS / Rs_km, proj5Pos_yPOS / Rs_km], 'k--', linewidth=2)
    
    plt.scatter(proj1Pos_xPOS / Rs_km, proj1Pos_yPOS / Rs_km, s=200, c='red', label=station1Name + ' Projection') # 'orange'
    plt.scatter(proj2Pos_xPOS / Rs_km, proj2Pos_yPOS / Rs_km, s=200, c='blue', label=station2Name + ' Projection') # 'green'
    # plt.scatter(proj3Pos_xPOS / Rs_km, proj3Pos_yPOS / Rs_km, s=50, c='cyan', label=station3Name + ' Projection')
    # plt.scatter(proj4Pos_xPOS / Rs_km, proj4Pos_yPOS / Rs_km, s=50, c='magenta', label=station4Name + ' Projection')
    # plt.scatter(proj5Pos_xPOS / Rs_km, proj5Pos_yPOS / Rs_km, s=50, c='purple', label=station5Name + ' Projection')
    plt.legend()
    plt.title('Plane of Sky @' + epochDt.strftime('%Y-%m-%d %H:%M:%S'))
    
    distance_12 = calculate_project_distance(proj1Pos_xPOS, proj1Pos_yPOS, proj2Pos_xPOS, proj2Pos_yPOS)
    print('distance (km) between '+station1Name+' and '+station2Name, distance_12)
    distance_13 = calculate_project_distance(proj1Pos_xPOS, proj1Pos_yPOS, proj3Pos_xPOS, proj3Pos_yPOS)
    print('distance (km) between '+station1Name+' and '+station3Name, distance_13)
    distance_23 = calculate_project_distance(proj2Pos_xPOS, proj2Pos_yPOS, proj3Pos_xPOS, proj3Pos_yPOS)
    print('distance (km) between '+station2Name+' and '+station3Name, distance_23)
    # distance_14 = calculate_project_distance(proj1Pos_xPOS, proj1Pos_yPOS, proj4Pos_xPOS, proj4Pos_yPOS)
    # print('distance (km) between '+station1Name+' and '+station4Name, distance_14)
    # distance_24 = calculate_project_distance(proj2Pos_xPOS, proj2Pos_yPOS, proj4Pos_xPOS, proj4Pos_yPOS)
    # print('distance (km) between '+station2Name+' and '+station4Name, distance_24)
    # distance_34 = calculate_project_distance(proj3Pos_xPOS, proj3Pos_yPOS, proj4Pos_xPOS, proj4Pos_yPOS)
    # print('distance (km) between '+station3Name+' and '+station4Name, distance_34)
    # distance_15 = calculate_project_distance(proj1Pos_xPOS, proj1Pos_yPOS, proj5Pos_xPOS, proj5Pos_yPOS)
    # print('distance (km) between '+station1Name+' and '+station5Name, distance_15)
    # distance_25 = calculate_project_distance(proj2Pos_xPOS, proj2Pos_yPOS, proj5Pos_xPOS, proj5Pos_yPOS)
    # print('distance (km) between '+station2Name+' and '+station5Name, distance_25)
    # distance_35 = calculate_project_distance(proj3Pos_xPOS, proj3Pos_yPOS, proj5Pos_xPOS, proj5Pos_yPOS)
    # print('distance (km) between '+station3Name+' and '+station5Name, distance_35)
    # distance_45 = calculate_project_distance(proj4Pos_xPOS, proj4Pos_yPOS, proj5Pos_xPOS, proj5Pos_yPOS)
    # print('distance (km) between '+station4Name+' and '+station5Name, distance_45)
    
    angle_12, radial_distance = calculate_angle_wrt_radial(proj1Pos_xPOS, proj1Pos_yPOS, proj2Pos_xPOS, proj2Pos_yPOS)
    print('radial distance (Rs) ', radial_distance)
    print('angle (deg.) between '+station1Name+' and '+station2Name, angle_12)
    angle_13, _ = calculate_angle_wrt_radial(proj1Pos_xPOS, proj1Pos_yPOS, proj3Pos_xPOS, proj3Pos_yPOS)
    print('angle (deg.) between '+station1Name+' and '+station3Name, angle_13)
    angle_23, _ = calculate_angle_wrt_radial(proj2Pos_xPOS, proj2Pos_yPOS, proj3Pos_xPOS, proj3Pos_yPOS)
    print('angle (deg.) between '+station2Name+' and '+station3Name, angle_23)
    # angle_14, _ = calculate_angle_wrt_radial(proj1Pos_xPOS, proj1Pos_yPOS, proj4Pos_xPOS, proj4Pos_yPOS)
    # print('angle (deg.) between '+station1Name+' and '+station4Name, angle_14)
    # angle_24, _ = calculate_angle_wrt_radial(proj2Pos_xPOS, proj2Pos_yPOS, proj4Pos_xPOS, proj4Pos_yPOS)
    # print('angle (deg.) between '+station2Name+' and '+station4Name, angle_24)
    # angle_34, _ = calculate_angle_wrt_radial(proj3Pos_xPOS, proj3Pos_yPOS, proj4Pos_xPOS, proj4Pos_yPOS)
    # print('angle (deg.) between '+station3Name+' and '+station4Name, angle_34)
    # angle_15, _ = calculate_angle_wrt_radial(proj1Pos_xPOS, proj1Pos_yPOS, proj5Pos_xPOS, proj5Pos_yPOS)
    # print('angle (deg.) between '+station1Name+' and '+station5Name, angle_15)
    # angle_25, _ = calculate_angle_wrt_radial(proj2Pos_xPOS, proj2Pos_yPOS, proj5Pos_xPOS, proj5Pos_yPOS)
    # print('angle (deg.) between '+station2Name+' and '+station5Name, angle_25)
    # angle_35, _ = calculate_angle_wrt_radial(proj3Pos_xPOS, proj3Pos_yPOS, proj5Pos_xPOS, proj5Pos_yPOS)
    # print('angle (deg.) between '+station3Name+' and '+station5Name, angle_35)
    # angle_45, _ = calculate_angle_wrt_radial(proj4Pos_xPOS, proj4Pos_yPOS, proj5Pos_xPOS, proj5Pos_yPOS)
    # print('angle (deg.) between '+station4Name+' and '+station5Name, angle_45)

    plt.gca().set_aspect(1)
    plt.xlabel('X (Rs)')
    plt.ylabel('Y (Rs)')
    plt.show()
