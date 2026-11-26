"""Infer effective solar geometry from TRAIN clear-sky diurnal shapes, without a gazetteer."""
import numpy as np
from scipy.optimize import least_squares

from features import RAD_CITIES
from round2_smooth import columns, row_interactions


def clock(dt, offset_hours):
    doy = dt.dt.dayofyear.to_numpy(dtype=float)
    phase = 2*np.pi*(doy-1)/365.25
    decl = (0.006918-0.399912*np.cos(phase)+0.070257*np.sin(phase)
            -0.006758*np.cos(2*phase)+0.000907*np.sin(2*phase)
            -0.002697*np.cos(3*phase)+0.00148*np.sin(3*phase))
    eq = 229.18*(0.000075+0.001868*np.cos(phase)-0.032077*np.sin(phase)
                 -0.014615*np.cos(2*phase)-0.040849*np.sin(2*phase))
    hour = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy()/60
    angle = (hour+offset_hours+eq/60-12)*np.pi/12
    return decl, angle


def infer_geometry(past):
    hr = past.loc[past.dt.dt.minute.eq(0)].reset_index(drop=True)
    # Deterministically sample complete days across the fitting history.
    hr = hr.loc[(np.arange(len(hr))//24) % 5 == 0]
    decl, hour = clock(hr.dt, 0.5)
    sd, cd = np.sin(decl), np.cos(decl)
    result = []
    for city in RAD_CITIES:
        clear = hr[f"{city}_clearsky"].to_numpy(dtype=float).reshape(-1,24)
        observed = clear/np.maximum(clear.max(axis=1,keepdims=True),1)
        def residual(params):
            latitude, longitude, exponent = params
            z = np.sin(latitude)*sd + np.cos(latitude)*cd*np.cos(hour+longitude)
            modeled = np.maximum(0,z.reshape(-1,24))**exponent
            return (modeled/np.maximum(modeled.max(1,keepdims=True),0.01) - observed).ravel()
        result.append(least_squares(residual, [np.pi/4,0,1.22],
                                    bounds=([0,-np.pi/4,0.7],[1.4,np.pi/4,2.4]),
                                    loss="soft_l1",f_scale=0.025).x)
    return np.array(result)


def projection_basis(frame, x, params):
    # Evaluate at the midpoint of the M2 half-hour window. Effective longitude
    # includes any systematic offset of the shipped hourly weather clock.
    dec, hour = clock(frame.dt, -0.25)
    lat, lon = params[:,0], params[:,1]
    sd, cd = np.sin(dec[:,None]), np.cos(dec[:,None])
    az = hour[:,None]+lon
    up = np.sin(lat)*sd+np.cos(lat)*cd*np.cos(az)
    south = np.sin(lat)*cd*np.cos(az)-np.cos(lat)*sd
    west = cd*np.sin(az)
    ghi = columns(x, "ghi_dec")/1000
    cs = columns(x, "clearsky_dec")/1000
    distance = 1+0.033*np.cos(2*np.pi*frame.dt.dt.dayofyear.to_numpy()/365.25)
    top = 1.361*distance[:,None]*np.maximum(up,0)
    kt = np.clip(ghi/np.maximum(top,0.035), 0, 1)
    diffuse_fraction = np.where(kt<=0.22, 1-0.09*kt, np.where(kt>0.80,0.165,
        0.9511-0.1604*kt+4.388*kt**2-16.638*kt**3+12.336*kt**4))
    diffuse = ghi*diffuse_fraction
    dni = np.minimum(1.2,(ghi-diffuse)/np.maximum(up,0.065))
    sky_fraction = np.clip(ghi/np.maximum(cs,0.025), 0, 1.15)
    # Alternative smooth beam fraction driven by shipped clear-sky irradiance.
    dni_clear = np.minimum(1.2,ghi*sky_fraction**2/np.maximum(up,0.065))
    temp = (columns(x, "temp_c_dec")-12)/15
    blocks = [diffuse, dni, dni_clear, np.maximum(up,0), ghi*up]
    seasonal = np.sin(dec)/0.40
    shape = np.column_stack([np.ones(len(frame)), seasonal, np.sin(hour), np.cos(hour),
                             seasonal*np.cos(hour), seasonal*np.sin(hour)])
    for tilt, bearing in [(20,0),(40,0),(65,0),(35,-65),(35,65)]:
        slope, orientation = np.deg2rad([tilt,bearing])
        incidence = np.maximum(0,up*np.cos(slope) + np.sin(slope)*(
            south*np.cos(orientation)+west*np.sin(orientation)))
        proj = dni*incidence
        blocks += [proj, dni_clear*incidence, proj*temp, row_interactions(proj.mean(1)[:,None],shape)]
    return np.column_stack(blocks)
