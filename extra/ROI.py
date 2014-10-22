#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This example shows how to use matplotlib to create regions of interest.  
I has two modes:
     - the segment mode, each time you click the mouse you can create a line 
segment 
     - the free hand mode, by keeping pressed the right button you can create a 
iregular contour 
The right button closes the loop.
Daniel Kornhauser
"""
from pylab import *

class ROI:
    
    def __init__(self, ax, fig):
        self.previous_point = []
        self.start_point = []
        self.end_point = []
        self.line = None    
        
        self.fig =  fig
        self.fig.canvas.draw()
        
    def motion_notify_callback(self, event):
        if event.inaxes: 
            ax = event.inaxes
            x, y = event.xdata, event.ydata
            if event.button == None and self.line != None: # Move line around 
                self.line.set_data([self.previous_point[0], x],
                                   [self.previous_point[1], y])      
                self.fig.canvas.draw()
            elif event.button == 1: # Free Hand Drawing
                    line = Line2D([self.previous_point[0], x],
                                  [self.previous_point[1], y])                  
  
                    ax.add_line(line)
                    self.previous_point = [x, y]
                    self.fig.canvas.draw()

        
    def button_press_callback(self, event):
        if event.inaxes: 
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            if event.button == 1:  # If you press the right button
                    if self.line == None: # if there is no line, create a line
                        self.line = Line2D([x,  x],
                                           [y, y],
                                           marker = 'o')
                        self.start_point = [x,y]
                        self.previous_point =  self.start_point 
                        ax.add_line(self.line)
                        self.fig.canvas.draw()
                    # add a segment
                    else: # if there is a line, create a segment
                        self.line = Line2D([self.previous_point[0], x], 
                                           [self.previous_point[1], y],
                                           marker = 'o')
                        self.previous_point = [x,y]
                        event.inaxes.add_line(self.line)
                        self.fig.canvas.draw()

            elif event.button == 3 and self.line != None: # close the loop
                        self.line.set_data([self.previous_point[0],self.start_point[0]],
                                           [self.previous_point[1],self.start_point[1]])                       
                        ax.add_line(self.line)
                        self.fig.canvas.draw()
                        self.line = None

def main():
    fig = figure()
    ax = fig.add_subplot(111)
    ax.set_title(" left click: line segment; left pressed: doodle; right click: close region")
    cursor = ROI(ax, fig)
    fig.canvas.mpl_connect('motion_notify_event', cursor.motion_notify_callback)
    fig.canvas.mpl_connect('button_press_event', cursor.button_press_callback)
    show()

if __name__ == "__main__":
    main()    
