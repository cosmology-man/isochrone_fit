# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:23:09 2022

@author: Asha
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sep
from matplotlib.patches import Ellipse
from astropy.modeling import models
from astropy import units as u
import math


print('initializing functions')
def openfile(folder, name, vmax_bruh, integration_time, wanted_integration_time):
    global_hdulist = []
    #opening and appending each fits 2D array to global_hdulist
    for i in range(len(os.listdir(folder))):
        global_hdulist.append((fits.open((str(folder)+str(os.listdir(folder)[i])))[0].data))
    #taking the median of all the images
    final_hdulist = np.median(np.array(global_hdulist), axis=0)
    #creating array of overscan
    overscan = final_hdulist[:, [1024, 1055]]
    #taking median of overscan
    median_overscan = np.median(overscan)
    #removing overscan from fits file
    final_hdulist -= median_overscan
    #scaling correctly for integration time
    final_hdulist *= wanted_integration_time/integration_time
    #deleting overscan
    end = np.delete(final_hdulist, np.arange(1024, 1056), axis = 1)
    #deleting error at column 256
    end = np.delete(end, 256, axis = 1)
    #deleting vignette in top and bottom left corners
    end = np.delete(end, np.arange(38), axis = 1)
    
    return end 

def objectdetect(name, final_science, extraction_number):
    #plt.figure(str(name) + 'background')
    #plt.title(str(name) + 'background')
    
    #creating background
    bkg = sep.Background(np.ascontiguousarray(final_science))
    #plt.imshow(bkg, interpolation='nearest', cmap='gray', origin='lower')
    #plt.colorbar()
    #plt.show()
    
    #subtracting background from image
    data_sub = np.ascontiguousarray(final_science) - bkg
    #extracting objects
    objects = sep.extract(data_sub, extraction_number, err=bkg.globalrms)
    #removing oblong objects
    delete = []
    for i in range(len(objects)):
        if objects[i]['a']/objects[i]['b'] > 1.25:
            delete.append(i)
    objects = np.delete(objects, delete)
    

    
    fig, ax = plt.subplots()
    m, s = np.mean(data_sub), np.std(data_sub)
    im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
            vmin=m-s, vmax=m+s, origin='lower')
    # plot an ellipse for each object
    
    plt.title(str(name))
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=6*objects['a'][i],
                    height=6*objects['b'][i], 
                    angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    return objects

def confide(R, B, V, num):
    objects = []
    for i in range(len(R)):
        red_x = R[i]['x']
        red_y = R[i]['y']
        for n in range(len(B)):
            blue_x = B[n]['x']
            blue_y = B[n]['y']
            if math.sqrt((red_x-blue_x)**2+(red_y-blue_y)**2) < num:
                for m in range(len(V)):
                    violet_x = V[m]['x']
                    violet_y = V[m]['y']
                    if math.sqrt((red_x-violet_x)**2+(red_y-violet_y)**2) < num:
                        if math.sqrt((violet_x-blue_x)**2+(violet_y-blue_y)**2) < num:
                            objects.append([R[i]['cflux'], B[n]['cflux'], V[m]['cflux']])
    return objects

def isochronefit(isochrone_file_path, HR_object):
    isochrone = np.loadtxt(str(isochrone_file_path), unpack=True)
    over = []
    for i in range(len(HR_object[0])):
        chi = []
        x = HR_object[0][i]
        y = HR_object[1][i]
        
        for n in np.linspace(7, 10, 61):
            tmp = []
            for q in range(len(isochrone[0])):
                if isochrone[0][q] != n:
                    tmp.append(q)
            temp = np.delete(isochrone, tmp, axis = 1)
            globe = []
            for q in range(len(temp[0])):
                globe.append(math.sqrt(((temp[-6][q]-temp[-7][q])-x)**2+(temp[-6][q]-y)**2))
            if globe == []:
                globe = globe
            else:
                chi.append(globe[globe.index(min(globe))])
        over.append(chi)
    return over

mzp = 20
def M15magnitudes(counts):
    mab=np.ndarray(len(counts))
    for i in range(len(counts)):
        N = counts[i]/120
        mi = (-2.5)*(math.log10(abs(N)))
        tempmab = mi + mzp
        mab[i] = tempmab
    return(mab)


def M29magnitudes(counts):
    mab=np.ndarray(len(counts))
    for i in range(len(counts)):
        N = counts[i]/60
        mi = (-2.5)*(math.log10(abs(N)))
        tempmab = mi + mzp
        mab[i] = tempmab
    return(mab)


print('producing science images')
#M29 flats
flat_R_M29 = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_R/', 'R Flats', 36000, 7, 60)
flat_B_M29 = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_B/', 'B Flats', 38000, 3, 60)
flat_V_M29 = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_V/', 'V Flats', 45000, 8, 60)

#M15 flats
flat_R_M15 = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_R/', 'R Flats', 36000, 7, 120)
flat_B_M15 = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_B/', 'B Flats', 38000, 3, 120)
flat_V_M15 = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_V/', 'V Flats', 45000, 8, 120)

#Calibration star flats
flat_B_Star = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_B/', 'B Flats', 38000, 3, 10)
flat_V_Star = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_V/', 'V Flats', 45000, 8, 10)
flat_R_Star = openfile('/users/asha/desktop/school/physics_136/l1d1/flat_R/', 'R Flats', 36000, 7, 2)


#M15 science files
M15_B = openfile('/users/asha/desktop/school/physics_136/l1d1/M15_B/', 'M15 B', 15000, 1, 1)
M15_R = openfile('/users/asha/desktop/school/physics_136/l1d1/M15_R/', 'M15 R', 55000, 1, 1)
M15_V = openfile('/users/asha/desktop/school/physics_136/l1d1/M15_V/', 'M15 V', 50000, 1, 1)

#M29 science files
M29_V = openfile('/users/asha/desktop/school/physics_136/l1d1/M29_V/', 'M29 V', 50000, 1, 1)
M29_R = openfile('/users/asha/desktop/school/physics_136/l1d1/M29_R/', 'M29 R', 50000, 1, 1)
M29_B = openfile('/users/asha/desktop/school/physics_136/l1d1/M29_B/', 'M29 B', 50000, 1, 1)

#Bias
Bias = openfile('/users/asha/desktop/school/physics_136/l1d1/Bias/', 'Bias', 30, 1, 1)

#Calibration star science files
Star_B = openfile('/users/asha/desktop/school/physics_136/l1d1/Star_B/', 'SZ Her B', 65000, 1, 1)
Star_V = openfile('/users/asha/desktop/school/physics_136/l1d1/Star_V/', 'SZ Her V', 65000, 1, 1)
Star_R = openfile('/users/asha/desktop/school/physics_136/l1d1/Star_R/', 'SZ Her R', 65000, 1, 1)

#subtracting bias from flats (mb idk why i did it like this)
flat_R_M15 -= Bias
flat_B_M15 -= Bias
flat_V_M15 -= Bias
flat_R_M29 -= Bias
flat_B_M29 -= Bias
flat_V_M29 -= Bias
flat_B_Star -= Bias
flat_V_Star -= Bias
flat_R_Star -= Bias

#subtracting bias from M15 science
M15_B -= Bias
M15_V -= Bias
M15_R -= Bias

#removing bias from M29 science
M29_V -= Bias
M29_R -= Bias
M29_B -= Bias

#removing bias from calibration star science 
Star_B -= Bias
Star_V -= Bias
Star_R -= Bias

#dividing M15 flats by their average
flat_R_M15 /= np.average(flat_R_M15)
flat_B_M15 /= np.average(flat_B_M15)
flat_V_M15 /= np.average(flat_V_M15)

#dividing M29 flats by their average
flat_R_M29 /= np.average(flat_R_M29)
flat_B_M29 /= np.average(flat_B_M29)
flat_V_M29 /= np.average(flat_V_M29)

#dividing calibration star by their average
flat_V_Star /= np.average(flat_V_Star)
flat_B_Star /= np.average(flat_B_Star)
flat_R_Star /= np.average(flat_R_Star)

#creating plots

"""
plt.figure('M29_V')
plt.imshow(M29_V/flat_V_M29, vmax = 6000, vmin = 0)
plt.title('M 29 final science image V')
plt.colorbar()
plt.show()

plt.figure('M29_B')
plt.imshow(M29_B/flat_B_M29, vmax = 900, vmin = 0)
plt.title('M 29 final science image B')
plt.colorbar()
plt.show()

plt.figure('M15_B')
plt.imshow(M15_B/flat_B_M15, vmax = 17000, vmin = 0)
plt.title('M 15 final science image B')
plt.colorbar()
plt.show()



plt.figure('M15_V')
plt.imshow(M15_V/flat_V_M15,vmax = 40000, vmin = 0)
plt.title('M 15 final science image V')
plt.colorbar()
plt.show()

plt.figure('Star B')
plt.imshow(Star_B/flat_B_Star, vmin = 0)
plt.title('Star B final science image V')
plt.colorbar()
plt.show()

plt.figure('Star V')
plt.imshow(Star_V/flat_V_Star, vmin = 0)
plt.title('Star B final science image V')
plt.colorbar()
plt.show()
"""



print('detecting objects...')
##Object Detection bruh
#M29
M29_B_objects = objectdetect('M29 B', M29_B/flat_B_M29, 3)
M29_R_objects = objectdetect('M29 R', M29_R/flat_R_M29, 3)
M29_V_objects = objectdetect('M29 V', M29_V/flat_V_M29, 3)


#Star
Star_B_objects = objectdetect('Star B', Star_B/flat_B_Star, 1)
Star_R_objects = objectdetect('Star R', Star_R/flat_R_Star, 1)
Star_V_objects = objectdetect('Star V', Star_V/flat_V_Star, 1)


#M15
M15_B_objects = objectdetect('M15 B', M15_B/flat_B_M15, 5)
M15_R_objects = objectdetect('M15 R', M15_R/flat_R_M15, 5)
M15_V_objects = objectdetect('M15 V', M15_V/flat_V_M15, 5)

M29_objects = confide(M29_B_objects, M29_V_objects, M29_R_objects, 3)
M15_objects = confide(M15_B_objects, M15_V_objects, M15_R_objects, 5)


B = []
V = []
R = []

"""
for i in M15_objects:
    B.append(i[0])
    V.append(i[1])
    R.append(i[2])
m, s = np.mean(M15_B/flat_B_M15), np.std(M15_B/flat_B_M15)
fig, ax = plt.subplots()
im = ax.imshow(M15_B/flat_B_M15, interpolation='nearest', cmap='gray',
            vmin=m-s, vmax=m+s, origin='lower')
for i in range(len(B)):
        e = Ellipse(xy=(B[i]['x'], B[i]['y']),
                    width=6*B[i]['a'],
                    height=6*B[i]['b'], 
                    angle=B[i]['theta'] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
plt.show()
"""
m, s = np.mean(M15_V/flat_V_M15), np.std(M15_V/flat_V_M15)
fig, ax = plt.subplots()
im = ax.imshow(M15_V/flat_V_M15, interpolation='nearest', cmap='gray',
            vmin=m-s, vmax=m+s, origin='lower')
for i in range(len(V)):
        e = Ellipse(xy=(V[i]['x'], V[i]['y']),
                    width=6*V[i]['a'],
                    height=6*V[i]['b'], 
                    angle=V[i]['theta'] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
plt.show()



print('producing HR diagrams...')
M29_objects = np.delete(M29_objects, 0, axis = 1)
M15_objects = np.delete(M15_objects, 0, axis = 1)

M29_B_V = M29_objects[:, 1]-M29_objects[:,0]
M15_B_V = M15_objects[:, 1]-M15_objects[:,0]

M29_B = np.delete(M29_objects, 0, axis = 1)
M15_B = np.delete(M15_objects, 0, axis = 1)


M29_B_V = M29magnitudes(M29_B_V)
M15_B_V = M15magnitudes(M15_B_V)

M29_B = M29magnitudes(M29_B)
M15_B = M15magnitudes(M15_B)



M29_HR = [M29_B_V, M29_B]
M15_HR = [M15_B_V, M15_B]


print('running fitting algorithm...')
#fitting isochrones
#M15 = isochronefit('/users/asha/downloads/isoc_z008s/isochrones/iso_jc_z008s.dat', M15_HR)
#final = np.sum(M15, axis = 1)

#print(np.argmin(final))


isochrone = np.loadtxt(str('/users/asha/downloads/isoc_z008s/isochrones/iso_jc_z008s.dat'), unpack=True)
tmp = []

for q in range(len(isochrone[0])):
    if isochrone[0][q] != 7.55:
        tmp.append(q)

temp = np.delete(isochrone, tmp, axis = 1)





