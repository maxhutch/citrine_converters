#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 07:39:31 2017

@author: bkappes
"""

#%% imports
import os, sys
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import patches
import json
import numpy as np

plt.rcParams['figure.figsize'] = (25, 9)
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

#%% set the working directory
WORKDIR = '/Users/bkappes/src/citrine_converters/docs/hough'
SRCFILE = 'P002_B001_G24.json'

os.chdir(WORKDIR)

#%% read data
with open(SRCFILE) as ifs:
    data = json.load(ifs)

strain = np.asarray([d['scalars'] for d in data['properties'] if d['name'] == 'strain'][0])
stress = np.asarray([d['scalars'] for d in data['properties'] if d['name'] == 'stress'][0])

#%% plot the stress-strain curve

def plot_ss(strain, stress):
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    _ = ax.plot(strain, stress, 'ro-')
    plt.show()
    
plot_ss(strain, stress)

#%% trim erratic behavior at high strain
for i in range(len(strain)):
    if strain[i] > 0.175:
        break
strain = strain[:i]
stress = stress[:i]

plot_ss(strain, stress)

#%% plot residual strain

def residual_strain(strain, stress, elastic_modulus):
    return strain - stress/elastic_modulus

approx_modulus = 325/0.00423

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
#ax.plot(strain, stress, 'ko')
#xlo, xhi = ax.get_xlim()
#ylo, yhi = ax.get_ylim()
for scaling in (1./4., 1./3., 1./2., 1.):
    elmod = scaling*approx_modulus
    residual = residual_strain(strain, stress, elmod)
    ax.plot(residual, stress, label='E = {:.0f} GPa'.format(elmod/1000))
ax.axvline(0, color='k', ls='-')
ax.set_xlabel(r'$\varepsilon_{residual}$')
ax.set_ylabel(r'$\sigma$')
plt.legend()
plt.savefig('residual-strain.png', dpi=300, bbox_inches='tight')


#%% resample
class Sampled():
    pass

sampled = Sampled()
sampled.strain = strain[::len(strain)//50]
sampled.stress = stress[::len(stress)//50]

plot_ss(sampled.strain, sampled.stress)

#%% standard form of a line

def standard_form(theta, xy):
    theta = np.radians(theta)
    x, y  = xy
    # ax + by + c = 0
    a = -np.sin(theta)
    b = np.cos(theta)
    c = -a*x - b*y
    return (a, b, c)

def point_line_distance(xy, abc):
    a, b, c = abc
    x, y = xy
    return np.abs(a*x + b*y + c)/np.sqrt(a**2 + b**2)

#%% plot alongside the accumulator

class Line(object):
    def __init__(self, abc):
        self.a, self.b, self.c = abc
    
    @property
    def abc(self):
        return (self.a, self.b, self.c)
    
    def xy(self, x=None, y=None):
        if x is None and y is None:
            raise ValueError('x or y must be specified')
        if x is None:
            x = (-self.b*y - self.c)/self.a
        if y is None:
            y = (-self.a*x - self.c)/self.b
        return (x, y)


def plot_with_accumulator(xvec, yvec, index, angle, accumulator=None):
    """Returns the accumulator"""
    # set the accumulator
    accumulator = accumulator if accumulator is not None else np.zeros((180, 180), dtype=int)
    # normalize the vectors
    xvec = (xvec - xvec.min())/(xvec.max() - xvec.min())
    yvec = (yvec - yvec.min())/(yvec.max() - yvec.min())
    # which point are we plotting
    x, y = xvec[index], yvec[index]
    # line that runs from origin to distance
    origin = Line(standard_form(angle + 90, (0, 0)))
#    print("origin: {}".format(origin.abc))
    # line that runs from the point of intersection
    xyline = Line(standard_form(angle, (x, y)))
#    print("xyline: {}".format(xyline.abc))
    # calculate distance
    qmax = 180.
    dmax = np.max(np.sqrt(xvec**2 + yvec**2))
    qstep = qmax / accumulator.shape[0]
    dstep = dmax / accumulator.shape[1]
    qi = int(angle / qstep)
    di = int(point_line_distance((0, 0), xyline.abc) / dstep)
#    print("(qstep, dstep) = ({}, {})".format(qstep, dstep))
#    print("(qi, di) = ({}, {})".format(qi, di))
    # increment accumulator
    accumulator[qi, di] += 1
    # plot
    gs = GridSpec(1, 2,
                  width_ratios=[16/9, 1])
    ax1 = plt.subplot(gs[:, :1])
    ax2 = plt.subplot(gs[:, 1:])
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    # plot x, y
    ax1.plot(xvec, yvec, 'ko-', label='function')
    xlo, xhi = ax1.get_xlim()
    ylo, yhi = ax1.get_ylim()
    ax1.axhline(0, color='k', ls='-')
    # plot the point of interest
    ax1.plot([x], [y], 'rd')
    # plot the distance line from the origin
    a1, b1, c1 = origin.abc
    a2, b2, c2 = xyline.abc
    dx, dy = np.dot(np.linalg.inv([[a1, b1], [a2, b2]]),
                    np.transpose([[-c1, -c2]])).T.ravel()
    ax1.arrow(0, 0, dx, dy, color='red', width=0.01, length_includes_head=True)
#    twidth = 0.1
#    xoff = twidth*np.cos(np.radians(angle + 90))
#    yoff = twidth*np.sin(np.radians(angle + 90))
#    ax1.text(dx/2 + xoff, dy/2 + yoff, 'd', fontsize=12)
    xlo, xhi = min(xlo, dx), max(xhi, dx)
    ylo, yhi = min(ylo, dy), max(yhi, dy)
    ax1.set_xlim(xlo, xhi)
    ax1.set_ylim(ylo, yhi)
    # plot the angle label
    if angle % 180 != 0:
        xint, yint = xyline.xy(y=0)
        a, b, c = xyline.abc
        # start of arc
        xA = 0.2 + (0 if c < 0 else xint)
        yA = 0
        mag = xA - xint
        # end of arc
        xB = np.cos(np.radians(angle))*mag + xint
        yB = np.sin(np.radians(angle))*mag + yint
        arrow = patches.FancyArrowPatch((xA, yA), (xB, yB),
                    color='red',
                    fill=True,
                    arrowstyle='fancy',
                    connectionstyle="angle3,angleA={},angleB={}".format(-90, angle-90))
        ax1.add_patch(arrow)
#        dx = np.cos(np.radians(angle/2))
#        dy = np.cos(np.radians(angle/2))
#        ax1.text((mag + twidth)*dx + xint, (mag + twidth)*dy + yint,
#                 r'$\theta$', fontsize=12)
#    print("(dx, dy) = ({}, {})".format(dx, dy))
    # plot the line that runs through point i
    xlo, xhi = ax1.get_xlim()
    ylo, yhi = ax1.get_ylim()
    xmin, ymin = xyline.xy(x=xlo)
    if ymin < ylo:
        xmin, ymin = xyline.xy(y=ylo)
    xmax, ymax = xyline.xy(x=xhi)
    if ymax > yhi:
        xmax, ymax = xyline.xy(y=yhi)
 #   print("(xmin, ymin) = ({}, {}))".format(xmin, ymin))
 #   print("(xmax, ymax) = ({}, {}))".format(xmax, ymax))
    ax1.plot([xmin, xmax], [ymin, ymax], 'r--')
    ax1.set_xlim(xlo, xhi)
    ax1.set_ylim(ylo, yhi)
    # plot the accumulator
    ax2.imshow(accumulator)
    ax2.set_xlabel('distance')
    ax2.set_ylabel(r'$\theta$')
    loc = ax2.get_xticks()
    lab = ["{:.2f}".format(float(l)*dstep) for l in loc]
    ax2.set_xticklabels(lab, rotation=75)
    loc = ax2.get_yticks()
    lab = ["{:.0f}".format(float(l)*qstep) for l in loc]
    ax2.set_yticklabels(lab)
    # additional annotations
    ax2.set_title('Hough transform', fontsize=18)
    # show
    plt.draw()
    return accumulator

acc = plot_with_accumulator(sampled.strain, sampled.stress, 36, 36)

#%% plot accumulator
acc = np.zeros((180, 180), dtype=int)
indexes = (18, 36)
for j in range(len(indexes)):
    for i in range(180):
        print("angle: {}".format(i))
        acc = plot_with_accumulator(sampled.strain, sampled.stress, indexes[j], i, acc)
        plt.savefig('frame-{:03d}.png'.format(j*180 + i))

