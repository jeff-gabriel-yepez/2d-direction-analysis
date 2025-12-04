#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: analysis.py
Author: Jeffrey G. Yepez
Date: 23 Jun 2025
Description: Analysis code for the direction algorithm paper.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from tqdm import tqdm

class Analysis():
    """
    This is a standalone code that can reproduce all of the figures in the submitted paper. This analysis code works solely with synthetic Gaussian distributions.
    """
    def __init__(self):
        # Control booleans.
        self.debug = False
        self.latex = True
        
        if self.debug:
            print("DataProcessor class method: __init__")

        if self.latex:
            plt.rc("font", family="serif", size = 18)
            plt.rcParams["text.usetex"] = True
            
            # Computer modern font.
            plt.rcParams["mathtext.fontset"] = "cm"

        return

    def rotateCoords(self, x_coords, y_coords, theta, plot=False):
        if self.debug:
            print("DataProcessor class method: rotateCoords")

        x_rot = []
        y_rot = []

        for x, y in zip(x_coords, y_coords):

            # Distance calculation in 2d.
            r = np.sqrt(x**2 + y**2)

            # Initial angle of capture.
            theta0 = np.arctan2(y, x)

            # Rotation calculation.
            phi = theta * np.pi / 180.0

            # Coordinate transformation.
            xprime = r * np.cos(theta0 - phi)
            yprime = r * np.sin(theta0 - phi)

            x_rot.append(xprime)
            y_rot.append(yprime)

        return x_rot, y_rot

    def CFND_analytical(self, theta, theta0, sigma, mu, seg_size):
        if self.debug:
            print("DataProcessor class method: CFND_analytical")

        norm_fac = 1 / (2 * np.pi * sigma**2)
        exponent = (mu**2 * (np.cos((theta0 + theta) * np.pi / 180.0) - 1)) / (2 * sigma**2)
        return seg_size * np.sqrt(norm_fac * (1 - np.exp(exponent)))

    def CFND_abs_sine_fit(self, theta, theta0, sigma, mu, seg_size):
        if self.debug:
            print("DataProcessor class method: CFND_abs_sine_fit")

        fac = seg_size * mu / ( np.sqrt(2 * np.pi) * sigma**2 )

        return fac * np.abs(np.sin((theta0 + theta) * np.pi / 180.0 / 2))


    def FNDAnalysis(self, save=False):
        if self.debug:
            print("DataProcessor class method: FNDAnalysis")

        # Number of counts to simulate.
        n = 10000

        grid_size = 16
        seg_size = 8

        l = (seg_size * grid_size) / 2.0

        sigma = 10
        mu_x = 2
        mu_y = 0

        # Mean of the Gaussian distribution and covariance matrix.
        mean = [mu_x, mu_y]  
        cov = [[sigma**2, 0], [0, sigma**2]]
        
        data = np.random.multivariate_normal(mean, cov, n)
        x, y = data[:, 0], data[:, 1]

        data_ref = np.random.multivariate_normal(mean, cov, n)
        x_ref, y_ref = data_ref[:, 0], data_ref[:, 1]


        # Define binning range
        x_range = (-l, l)
        y_range = (-l, l)

        # Create a 2D histogram with binned data over a specific range
        hist, xedges, yedges = np.histogram2d(y, x, bins=grid_size, range=[x_range, y_range])
        ref, xedges_ref, yedges_ref = np.histogram2d(y_ref, x_ref, bins=grid_size, range=[x_range, y_range])

        angles = range(-180, 180)
        norms = []

        ref = ref / np.sum(ref)
        
        # Calculate FND of simulated data.
        for theta in tqdm(angles):
            x_rot, y_rot = self.rotateCoords(x, y, theta)
            
            rot, xedges_rot, y_edges_rot = np.histogram2d(y_rot, x_rot, bins=grid_size, range=[x_range, y_range])
            rot = rot / np.sum(rot)

            FND = np.sqrt(np.sum(np.square(ref - rot)))
            
            norms.append(FND)

        plt.plot(angles, norms, "r.", label="FND of Gaussian data")
        plt.plot(angles, self.CFND_analytical(np.array(angles), 0, sigma, np.sqrt(mu_x**2 + mu_y**2), seg_size), color="black")
        
        plt.xticks(ticks=[-np.pi*180/np.pi,-np.pi/2*180/np.pi,0,np.pi/2*180/np.pi,np.pi*180/np.pi], labels=["$-\\pi$","$-\\pi/2$","$0$", "$-\\pi/2$", "$\\pi$"])
        
        plt.xlabel("$\\vartheta$")
        plt.ylabel("FND")

        if save:
            plt.savefig(f"sim_n_{n}.pdf", format="pdf", bbox_inches="tight")
            
        plt.show()

        return

    def FNDAnalysisSampGauss(self, save=False):
        if self.debug:
            print("DataProcessor class method: FNDAnalysisSampGauss")

        A = 10

        grid_size = 128

        seg_size = 1

        l = (seg_size * grid_size) / 2.0

        sigma = 10
        mu_x = 2
        mu_y = 0

        # Plot continuous 2d Gaussian distribution.
        n_points = grid_size

        # Create a grid of points in the specified region.
        x = np.linspace(-l, l, n_points)
        y = np.linspace(-l, l, n_points)

        # 2D grid of points.
        X, Y = np.meshgrid(x, y)

        # Compute the values of the function at each point in the grid.
        Z = self.gaussian_2d((X, Y), A, mu_x, mu_y, sigma, sigma)

        # Create the figure and polar axes.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        img = ax.imshow(Z, extent=[-l, l, -l, l], origin="lower", cmap="gray_r", aspect="equal")

        img.set_clim(vmin=0, vmax=100)
        
        cbar = fig.colorbar(img, label="Function value")

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax.set_title("Sampled Gaussian distribution")
        
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)

        if save:
            plt.savefig(f"c_normal_dist_fit_plot.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        angles = range(-180, 180)
        norms = []

        ref = self.sym_2d_norm_dist_amp(X, Y, 0, sigma, np.sqrt(mu_x**2 + mu_y**2), A)
        ref = ref / np.sum(ref)
        for theta in angles:            
            hist = self.sym_2d_norm_dist_amp(X, Y, theta * np.pi/180.0, sigma, np.sqrt(mu_x**2 + mu_y**2), A)
            hist = hist / np.sum(hist)

            FND = np.sqrt(np.sum(np.square(ref - hist)))

            norms.append(FND)

        plt.plot(angles, norms, "r.", label="FND of sampled Gaussian")
        plt.plot(angles, self.CFND_abs_sine_fit(np.array(angles), 0, sigma, np.sqrt(mu_x**2 + mu_y**2), seg_size), "k")
        
        plt.xlabel("$\\vartheta$")
        plt.ylabel("FND")
        
        plt.xticks(ticks=[-np.pi*180/np.pi,-np.pi/2*180/np.pi,0,np.pi/2*180/np.pi,np.pi*180/np.pi], labels=["$-\\pi$","$-\\pi/2$","$0$", "$-\\pi/2$", "$\\pi$"])

        if save:
            plt.savefig(f"sim_deltax_{seg_size}.pdf", format="pdf", bbox_inches="tight")
            
        plt.show()

        return

    def gaussian(self, x, amp, mu, sigma):
        if self.debug:
            print("DataProcessor class method: gaussian")

        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def normal(self, x, amp, mu, sigma):
        if self.debug:
            print("DataProcessor class method: normal")

        return (1/(np.sqrt(2*np.pi)*np.abs(sigma))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def gaussian_2d(self, coords, A, mu_x, mu_y, sigma_x, sigma_y):
        if self.debug:
            print("DataProcessor class method: gaussian_2d")

        x, y = coords
        return A * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))

    def sym_2d_norm_dist_amp(self, x, y, theta, sigma, r, A):
        if self.debug:
            print("DataProcessor class method: sym_2d_norm_dist_amp")

        return A * np.exp(-(x**2 + y**2 + r**2) / (2 * sigma**2)) * np.exp(-(r * (x * np.cos(theta) + y * np.sin(theta))) / (sigma**2))

    def gaussHist1D(self, save=False):
        if self.debug:
            print("DataProcessor class method: gaussHist1D")

        np.random.seed(259)
        data = np.random.normal(loc=0, scale=1, size=1000)

        # Create histogram.
        bins = 32
        hist_vals, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fit Gaussian function to histogram.
        popt, _ = curve_fit(self.gaussian, bin_centers, hist_vals, p0=[1, 5, 2])

        print(popt)

        # Plot histogram and fitted curve.
        fig, ax1 = plt.subplots()
        ax1.hist(data, bins=bins, label="Data", histtype="step", edgecolor="black", linewidth=2)
        x_vals = np.linspace(min(data), max(data), 1000)
        ax1.plot(x_vals, self.gaussian(x_vals, *popt), "k--", linewidth=2, label="Gaussian fit")
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("Counts")
        ax1.set_xlim(-4,4)
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper right")

        ax1.axvline(0, linestyle="--", color="gray")
        ax1.annotate("$\\mu=0$",
                     xy=(0.14, 7))

        ax2 = ax1.twinx()

        ax2.plot(x_vals, self.normal(x_vals, *popt), "-", color="mediumblue", linewidth=2)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("Normal distribution", color="mediumblue")
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(0, 1)

        ax2.tick_params(axis="y", labelcolor="mediumblue")

        if save:
            plt.savefig("gauss_fit_normal_combined.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        return

    def gaussHist2D(self, save=False):
        if self.debug:
            print("DataProcessor class method: gaussHist2D")

        # Parameters.
        mu_x, mu_y = 0, 0
        sigma_x, sigma_y = 1, 1

        # Create grid.
        x = np.linspace(-4, 4, 16)
        y = np.linspace(-4, 4, 16)
        X, Y = np.meshgrid(x, y)

        # Calculate 2D Gaussian values
        Z = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
            -((X - mu_x)**2 / (2 * sigma_x**2) + (Y - mu_y)**2 / (2 * sigma_y**2))
        )

        # Prepare data for histogram.
        xpos = X.ravel()
        ypos = Y.ravel()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.5
        dz = Z.ravel()

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color="skyblue", edgecolor="black")

        # Adjust the elevation and azimuth for a nice view.
        ax.view_init(elev=20, azim=30)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Normal distribution")

        if save:
            plt.savefig("LEGO_hist.pdf", format="pdf")

        plt.tight_layout()
        plt.show()

        return

    def gaussPlot2D(self, save=False):
        if self.debug:
            print("DataProcessor class method: gaussPlot2D")

        # Gaussian parameters.
        mu_x = 0
        mu_y = 0
        sigma_x = 1
        sigma_y = 1

        # Create a grid of (x, y) values.
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)

        # 2D Gaussian function.
        Z = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
            -(((X - mu_x) ** 2) / (2 * sigma_x ** 2) + ((Y - mu_y) ** 2) / (2 * sigma_y ** 2))
        )

        # Plotting.
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Surface plot
        s = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        
        # Labels and viewing angle.
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.view_init(elev=20, azim=30)  # Adjust the elevation and azimuth for a nice view

        cbar = fig.colorbar(s, shrink=0.6, aspect=14, pad=-0.00, label="Normal distribution")
        s.set_clim(0-0.01,0.15+0.01)
        ax.zaxis.set_major_formatter("")

        ax.set_zticks([0,0.05,0.1,0.15])

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        plt.tight_layout()

        if save:
            plt.savefig("3d_gauss.pdf", format="pdf")

        plt.show()

        return

    def plotRotatedBinnedGauss(self, save=False):
        if self.debug:
            print("DataProcessor class method: plotRotatedBinnedGauss")

        grid_size = 3

        seg_size = 24

        n = 10000

        l = (seg_size * grid_size) / 2.0

        sigma = 10

        t = 30
        mu = 5
        
        mu_x = mu * np.cos(-t * np.pi / 180.0)
        mu_y = mu * np.sin(-t * np.pi / 180.0)

        # Mean of the Gaussian distribution and covariance.
        mean = [mu_x, mu_y]
        cov = [[sigma**2, 0], [0, sigma**2]]
        data = np.random.multivariate_normal(mean, cov, n)
        x, y = data[:, 0], data[:, 1]

        data_ref = np.random.multivariate_normal(mean, cov, n)
        x_ref, y_ref = data_ref[:, 0], data_ref[:, 1]

        # Define binning range.
        x_range = (-l, l)
        y_range = (-l, l)

        # Create a 2D histogram with binned data over a specific range.
        hist, xedges, yedges = np.histogram2d(y, x, bins=grid_size, range=[x_range, y_range])

        high_res, xedges_res, yedges_res = np.histogram2d(y_ref, x_ref, bins=128, range=[x_range, y_range])
        
        ref, xedges_ref, yedges_ref = np.histogram2d(y_ref, x_ref, bins=grid_size, range=[x_range, y_range])

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()

        
        img = ax.imshow(high_res, cmap="Blues")
        img.set_clim(vmin=0, vmax=8)

        ax.set_axis_off()

        if save:
            plt.savefig(f"gauss_data_theta{t}.pdf", bbox_inches="tight", pad_inches=0)        
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()


        img = ax.imshow(ref, cmap="Blues")
        img.set_clim(vmin=0, vmax=2000)

        for i in range(grid_size):
            for j in range(grid_size):
                text = plt.text(j, i, f"{int(ref[i, j])}", 
                                ha="center", va="center", color=("w" if int(ref[i, j]) >= 1200 else "k"))

        ax.set_axis_off()

        if save:
            plt.savefig(f"gauss_bins_theta{t}.pdf", bbox_inches="tight", pad_inches=0)        

        plt.show()

        return

    def plotDifference(self, save=False):
        if self.debug:
            print("DataProcessor class method: plotDifference")

        A = 10

        grid_size = 32

        seg_size = 2

        l = (seg_size * grid_size) / 2.0

        r = 10

        sigma = 10
        mu_x = r
        mu_y = 0

        # Plot continuous 2d Gaussian distribution.
        n_points = grid_size

        # Create a grid of points in the specified region.
        x = np.linspace(-l, l, n_points)
        y = np.linspace(-l, l, n_points)

        # 2D grid of points.
        X, Y = np.meshgrid(x, y)

        # Compute the values of the function at each point in the grid.
        Z1 = self.gaussian_2d((X, Y), A, mu_x, mu_y, sigma, sigma)

        #vm = 5
        #vm = 5
        vm = 20
        
        #t = 0
        #t = 15
        t = 45

        mu_x = r * np.cos(t * np.pi / 180)
        mu_y = r * np.sin(t * np.pi / 180)

        # Plot continuous 2d Gaussian distribution.
        n_points = grid_size

        # Create a grid of points in the specified region.
        x = np.linspace(-l, l, n_points)
        y = np.linspace(-l, l, n_points)

        # 2D grid of points.
        X, Y = np.meshgrid(x, y)

        # Compute the values of the function at each point in the grid.
        Z2 = self.gaussian_2d((X, Y), A, mu_x, mu_y, sigma, sigma)

        Z = (Z1 - Z2)**2

        # Create the figure.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        img = ax.imshow(Z, extent=[-l, l, -l, l], origin="lower", cmap="viridis", aspect="equal")

        img.set_clim(vmin=0, vmax=vm)
        
        cbar = fig.colorbar(img)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)

        if save:
            plt.savefig(f"2d_gauss_diff_plot_t{t}.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        return

if __name__ == "__main__":
    """
    This is the main statement where the user can execute subroutines.
    """
    a = Analysis()

    # These are the subroutines to run to produce all of the figures in the paper. The parameters for each subroutine are located within the subroutine code.
    a.FNDAnalysis()
    a.gaussHist2D()
    a.gaussHist1D()
    a.FNDAnalysisSampGauss()
    a.gaussPlot2D()
    a.plotDifference()
    a.plotRotatedBinnedGauss()

    sys.exit()
        
