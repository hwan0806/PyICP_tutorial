from tkinter import PROJECTING
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlotTrajectory:
    def __init__(self,x,y,z):
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        print(x)
        
        ax.plot(x,y,z)
        