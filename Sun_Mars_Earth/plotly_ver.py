import PIL.Image
import numpy as np
import spiceypy as spice
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.offline

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
    stationPosHelioCentric, _ = spice.spkcpt(stationPosItrs, 'EARTH BARYCENTER', 'ITRF93', epochEt, coord, 'OBSERVER',
                                             'NONE', 'SUN')
    return stationPosHelioCentric[:3]


def plot_SME(startDt, endDt, stepDt, POS_type='EM'):
    epochDt = create_epoch([startDt, endDt], stepDt)
    earthPos = np.array(get_body_pos('EARTH', epochDt, ))
    marsPos = np.array(get_body_pos('MARS BARYCENTER', epochDt))
    # stationPos = get_station_pos('SH',epochDt,)
    if POS_type == 'EM':
        vecPOSn = np.array((earthPos - marsPos) / np.linalg.norm((earthPos - marsPos).T))
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
    plt.gca().add_patch(plt.Circle((0, 0), 696300. / AU_km, linewidth=1, color='r'))
    plt.plot(earthPos[:, 0] / AU_km, earthPos[:, 1] / AU_km, c='b', label='Earth')
    plt.plot(marsPos[:, 0] / AU_km, marsPos[:, 1] / AU_km, c='orange', label='Mars')
    plt.plot(projPos[:, 0] / AU_km, projPos[:, 1] / AU_km, c='k', label='Projections')
    for i in range(len(epochDt)):
        plt.plot([earthPos[i, 0] / AU_km, marsPos[i, 0] / AU_km], [earthPos[i, 1] / AU_km, marsPos[i, 1] / AU_km],
                 linewidth=1., c='gray')
        plt.plot([0, projPos[i, 0] / AU_km], [0, projPos[i, 1] / AU_km], linewidth=1., c='pink')
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


def get_sphere(R0, lon, lat, for_PSI=False):
    if for_PSI:
        lat = np.pi / 2 - lat
    llon, llat = np.meshgrid(lon, lat)
    x0 = R0 * np.cos(llon) * np.cos(llat)
    y0 = R0 * np.sin(llon) * np.cos(llat)
    z0 = R0 * np.sin(llat)
    return x0, y0, z0


def plot_planet_model(planet_str, dt, radius=5., transform=False):
    et = spice.datetime2et(dt)
    planet_pos, _ = spice.spkpos(planet_str, et, 'IAU_SUN', 'NONE', 'SUN')
    planet_pos = np.array(planet_pos).T
    if transform:
        trans_arr = spice.sxform('IAU_EARTH', 'IAU_SUN', et)

    # planet_origin, _ = spice.spkcpt([radius,0.,0.], 'EARTH BARYCENTER', 'ITRF93', et, 'IAU_SUN', 'OBSERVER',
    #                                          'NONE', 'EARTH BARYCENTER')
    # planet_origin = np.array(planet_origin[0:3]).T #- planet_pos
    # earth_radii = spice.bodvrd('EARTH', 'RADII', 3)
    # earth_re = earth_radii[1][0]
    # earth_rp = earth_radii[1][2]
    # earth_f = (earth_re - earth_rp) / earth_re
    # planet_origin_lon, planet_origin_lat, _= spice.recgeo(planet_origin, earth_radii[0],
    #                                                             earth_f)
    # print(planet_origin_lon,planet_origin_lat)

    import matplotlib.pyplot as plt
    img = plt.imread('Planet_model/' + planet_str + '.jpeg',format='grayscale')
    from PIL import Image, ImagePalette
    eight_bit_img = Image.fromarray(img)#.convert('P',
                                                            # palette=PIL.Image.Palette(0),
                                                            # dither=None,
                                                            # )

    # idx_to_color = np.array(eight_bit_img.getpalette()).reshape(-1, 3)
    # colorscale = [[i / 255.0, 'rgb({},{},{})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    texture = np.asarray(Image.open('Planet_model/' + planet_str + '.jpeg').convert('L'))
    if planet_str == 'EARTH BARYCENTER':
        colorscale = [[0.0, 'rgb(20, 20, 200)'],
                      [0.1, 'rgb(30, 30, 190)'],
                      [0.2, 'rgb(70, 70, 170)'],

                      [0.3, 'rgb(115,141,90)'],
                      [0.4, 'rgb(122, 126, 75)'],

                      [0.6, 'rgb(122, 126, 75)'],
                      [0.7, 'rgb(141,115,96)'],
                      [0.8, 'rgb(223, 197, 170)'],
                      [0.9, 'rgb(237,214,183)'],

                      [1.0, 'rgb(255, 255, 255)']]
    elif planet_str == 'MARS BARYCENTER':
        colorscale = [[0.0, 'rgb(20,20,20)'],
                      [0.1, 'rgb(100,30,30)'],
                      [0.2, 'rgb(148,40,40)'],

                      [0.3, 'rgb(190,91,60)'],
                      [0.4, 'rgb(200, 100, 70)'],

                      [0.6, 'rgb(220, 120, 75)'],
                      [0.7, 'rgb(230,130,80)'],
                      [0.8, 'rgb(232,140,95)'],
                      [0.9, 'rgb(250,200,200)'],

                      [1.0, 'rgb(255, 255, 255)']]
    elif planet_str == 'SUN':
        colorscale = [[0.0, 'rgb(30,0,0)'],
                      [0.1, 'rgb(50,10,10)'],
                      [0.2, 'rgb(70,20,20)'],

                      [0.3, 'rgb(100,30,30)'],
                      [0.4, 'rgb(130, 35, 35)'],

                      [0.6, 'rgb(200, 40, 40)'],
                      [0.7, 'rgb(230,130,80)'],
                      [0.8, 'rgb(240,140,95)'],
                      [0.9, 'rgb(250,200,200)'],

                      [1.0, 'rgb(255, 255, 255)']]

    lon = np.linspace(-np.pi, np.pi, img.shape[1])
    lat = np.linspace(np.pi / 2, -np.pi / 2, img.shape[0])
    x0, y0, z0 = get_sphere(radius, lon, lat)
    if transform:
        for j in range(len(lon)):
            for i in range(len(lat)):
                x0[i, j], y0[i, j], z0[i, j] = np.dot(trans_arr[0:3, 0:3], [x0[i, j], y0[i, j], z0[i, j]])

    print('Plotting ' + planet_str)
    trace = go.Surface(x=x0 + planet_pos[0],
                       y=y0 + planet_pos[1],
                       z=z0 + planet_pos[2],
                       surfacecolor=texture,cmin=0,cmax=255,colorscale=colorscale, showscale=True)
    return trace


if __name__ == '__main__':
    df = pd.read_csv('coordinate_list.txt', sep='\s+', header=None, names=['StationName', 'Number', 'x', 'y', 'z'])
    earth_radii = spice.bodvrd('EARTH', 'RADII', 3)
    earth_re = earth_radii[1][0]
    earth_rp = earth_radii[1][2]
    earth_f = (earth_re - earth_rp) / earth_re
    df['x'] /= 1000.
    df['y'] /= 1000.
    df['z'] /= 1000.
    # %%
    # fig_earth = go.Figure()
    # lon = np.linspace(-np.pi, np.pi, 360)
    # lat = np.linspace(np.pi / 2, -np.pi / 2, 180)
    # x0, y0, z0 = get_sphere(earth_radii[1][0], lon, lat)
    # fig_earth.add_trace(go.Surface(x=x0, y=y0, z=z0))
    # fig_earth.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['z']))
    # plotly.offline.plot(fig_earth)
    # %%
    df['lon'], df['lat'], df['alt'] = 0., 0., 0.
    for i in range(len(df)):
        df['lon'][i], df['lat'][i], df['alt'][i] = spice.recgeo([df['x'][i], df['y'][i], df['z'][i]], earth_radii[0],
                                                                earth_f)
    df['lon'] = np.rad2deg(df['lon'])
    df['lat'] = np.rad2deg(df['lat'])

    # fig_geo = go.Figure()
    #
    # fig_geo.add_trace(go.Scattergeo(lon=df['lon'], lat=df['lat'], text=df['StationName'],
    #                                 mode='markers+text', textposition='top center',
    #                                 marker=dict(size=10)))
    # # fig_geo.update_geos(projection_type='orthographic')
    #
    # fig_geo.update_layout(title='Stations', geo_scope='world')
    #
    # plotly.offline.plot(fig_geo)

    # %%
    from datetime import datetime, timedelta

    epochDt = datetime(2021, 10, 4, 0, 0, 0)
    earthPos = np.array(get_body_pos('EARTH BARYCENTER', epochDt, ))
    marsPos = np.array(get_body_pos('MARS BARYCENTER', epochDt))
    vecPOSn = np.array(earthPos / np.linalg.norm(earthPos.T))
    vecPOSx = np.cross([0, 0, 1], vecPOSn)
    vecPOSx = vecPOSx / np.linalg.norm(vecPOSx)
    vecPOSy = np.cross(vecPOSn, vecPOSx)
    vecPOSy = vecPOSy / np.linalg.norm(vecPOSy)

    OM = marsPos
    ME = earthPos - marsPos
    Nvec = vecPOSn
    result_df = pd.DataFrame()
    df['S_in_HG_x'], df['S_in_HG_y'], df['S_in_HG_z'] = 0., 0., 0.
    df['P_in_HG_x'], df['P_in_HG_y'], df['P_in_HG_z'] = 0., 0., 0.
    df['P_on_POS_x'], df['P_on_POS_y'] = 0., 0.
    df['visible'] = 0
    df['Mars_angle_deg'] = 0.
    for i in range(len(df)):
        stationName = df['StationName'][i]
        print(stationName)
        stationPos = get_station_pos(stationName, epochDt)
        df['S_in_HG_x'][i], df['S_in_HG_y'][i], df['S_in_HG_z'][i] = stationPos[0], stationPos[1], stationPos[2]
        projPos = (np.dot(stationPos, Nvec) * OM - np.dot(OM, Nvec) * stationPos) / (
                np.dot(stationPos, Nvec) - np.dot(OM, Nvec))
        projPos_xPOS, projPos_yPOS = np.dot(projPos, vecPOSx), np.dot(projPos, vecPOSy)
        # result_df._append({'Name':stationName,'P_in_HG_x': projPos[0], 'P_in_HG_y': projPos[1],'P_in_HG_z': projPos[2],
        #                   'P_on_POS_x':projPos_xPOS,'P_on_POS_y':projPos_yPOS},ignore_index=True)
        df['P_in_HG_x'][i], df['P_in_HG_y'][i], df['P_in_HG_z'][i], df['P_on_POS_x'][i], df['P_on_POS_y'][i] = projPos[
            0], projPos[1], projPos[2], projPos_xPOS, projPos_yPOS
        ES = stationPos - earthPos
        df['Mars_angle_deg'][i] = np.rad2deg(np.arccos(np.dot(ES,-ME)/(np.linalg.norm(ES)*np.linalg.norm(ME))))
        if np.dot(ES, ME) < 0:
            df['visible'][i] = 1
        # df_tmp = pd.DataFrame(
        #     {'Name': stationName, 'P_in_HG_x': projPos[0], 'P_in_HG_y': projPos[1], 'P_in_HG_z': projPos[2],
        #      'P_on_POS_x': projPos_xPOS, 'P_on_POS_y': projPos_yPOS}, index=[0])
        # result_df = pd.concat([result_df, df_tmp], ignore_index=True)

    fig_3d = go.Figure()
    fig_3d.add_trace(plot_planet_model('EARTH BARYCENTER', epochDt, radius=6300, transform=True))
    fig_3d.add_trace(plot_planet_model('MARS BARYCENTER', epochDt, radius=3390))
    fig_3d.add_trace(plot_planet_model('SUN', epochDt, radius=696300))
    for i in range(len(df)):
        if df['visible'][i]:
            fig_3d.add_trace(go.Scatter3d(x=[df['S_in_HG_x'][i], df['P_in_HG_x'][i], marsPos[0]],
                                          y=[df['S_in_HG_y'][i], df['P_in_HG_y'][i], marsPos[1]],
                                          z=[df['S_in_HG_z'][i], df['P_in_HG_z'][i], marsPos[2]],
                                          mode='markers+lines+text',
                                          text=df['StationName'][i]+' (Mars Zenith Angle=%.2f deg)'%df['Mars_angle_deg'][i], #' (Mars Zenith Angle=%.2f deg)'%df['Mars_angle_deg'][i],#,' Projection @ Plane of Sky'
                                          textposition='middle right'
                                          ))
            # fig_3d.add_trace(go.Scatter3d(x=[df['S_in_HG_x'][i]],
            #                               y=[df['S_in_HG_y'][i]],
            #                               z=[df['S_in_HG_z'][i]],
            #                               mode='markers+text',
            #                               text='Stations @ Earth',
            #                               textposition='top left'
            #                               ))
            # fig_3d.add_trace(go.Scatter3d(x=[df['P_in_HG_x'][i]],
            #                               y=[df['P_in_HG_y'][i]],
            #                               z=[df['P_in_HG_z'][i]],
            #                               mode='markers+text',
            #                               text='Projections @ Plane of Sky',
            #                               textposition='top left'
            #                               ))
            # fig_3d.add_trace(go.Scatter3d(x=[marsPos[0]],
            #                               y=[marsPos[1]],
            #                               z=[marsPos[2]],
            #                               mode='markers+text',
            #                               text='Mars',
            #                               textposition='top left'
            #                               ))
            fig_3d.add_trace(go.Scatter3d(x=[0., df['P_in_HG_x'][i]],
                                          y=[0., df['P_in_HG_y'][i]],
                                          z=[0., df['P_in_HG_z'][i]],
                                          mode='lines', line=dict(color='pink')))
    center_pos = [df['P_in_HG_x'][0], df['P_in_HG_y'][0], df['P_in_HG_z'][0]]
    center_pos = earthPos
    # center_pos = [0,0,0]
    edge1_pos = (vecPOSx + vecPOSy) * 100 * Rs_km
    edge2_pos = (-vecPOSx + vecPOSy) * 100 * Rs_km
    edge3_pos = (vecPOSx - vecPOSy) * 100 * Rs_km
    edge4_pos = (-vecPOSx - vecPOSy) * 100 * Rs_km

    fig_3d.add_trace(go.Mesh3d(x=[edge1_pos[0], edge2_pos[0], edge3_pos[0], edge4_pos[0]],
                               y=[edge1_pos[1], edge2_pos[1], edge3_pos[1], edge4_pos[1]],
                               z=[edge1_pos[2], edge2_pos[2], edge3_pos[2], edge4_pos[2]],
                               opacity=0.5, color='azure'))

    fig_3d.update_layout(scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        # xaxis_range=[earthPos[0] - 10000, earthPos[0] + 10000],
        # yaxis_range=[earthPos[1] - 10000, earthPos[1] + 10000],
        # zaxis_range=[earthPos[2] - 10000, earthPos[2] + 10000],
        xaxis_range=[center_pos[0] - 10000, center_pos[0] + 10000],
        yaxis_range=[center_pos[1] - 10000, center_pos[1] + 10000],
        zaxis_range=[center_pos[2] - 10000, center_pos[2] + 10000],
        ),
        template='plotly_dark',
        scene_aspectratio=dict(x=1, y=1, z=1),
    )
    plotly.offline.plot(fig_3d)
