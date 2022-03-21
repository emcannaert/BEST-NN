#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# imageOperations.py --------------------------------------------------------------
#==================================================================================
# This module contains functions to make Jet Images -------------------------------
#==================================================================================

# modules
import numpy
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import copy
import random
import itertools
import types
import tempfile
import sys
# grab some keras stuff
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
import keras.backend as K


#==================================================================================
# Plot Averaged Boosted Jet Images ------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotAverageBoostedJetImage(jetImageDF, title, plotPNG, plotPDF):

   # sum and average jet images
   summed = numpy.sum(jetImageDF, axis=0)
   avg = numpy.apply_along_axis(lambda x: x/len(jetImageDF), axis=1, arr=summed)

   # plot the images
   plt.figure('N') 
   plt.imshow(avg[:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', extent=[-numpy.pi, numpy.pi, -1, 1], aspect = "auto")
   cbar = plt.colorbar()
   cbar.set_label(r'Energy [GeV]')
   if title == 'boost_QCD' :
      plt.title('QCD Boosted Jet Image', fontsize = 22)
   if title == 'boost_HH4W' :
      plt.title(r'$H\rightarrow WW$ Boosted Jet Image', fontsize = 22)
   if title == 'boost_HH4B' :
      plt.title(r'$H\rightarrow bb$ Boosted Jet Image', fontsize = 22)
   plt.xlabel(r'$\phi$', fontsize = 18)
   plt.ylabel(r'cos($\theta$)', fontsize = 20)
   if plotPNG == True :
      plt.savefig('plots/'+title+'_jetImage.png')
   if plotPDF == True :
      plt.savefig('plots/'+title+'_jetImage.pdf')
   plt.close()

#==================================================================================
# Plot 3 Boosted Jet Images -------------------------------------------------------
#----------------------------------------------------------------------------------
# Average over the jet images and plot the result as a 2D histogram ---------------
# title has limited options, see if statements ------------------------------------
#----------------------------------------------------------------------------------

def plotThreeBoostedJetImages(jetImageDF, title, plotPNG, plotPDF):

   for i in range (0, 3) :
      # plot the images
      plt.figure('N') 
      plt.imshow(jetImageDF[i,:,:,0].T, norm=mpl.colors.LogNorm(), origin='lower', interpolation='none', extent=[-numpy.pi, numpy.pi, -1, 1], aspect = "auto")
      cbar = plt.colorbar()
      cbar.set_label(r'Energy [GeV]')
      if title == 'boost_QCD' :
         plt.title('QCD Boosted Jet Image #'+str(i), fontsize = 22)
      if title == 'boost_HH4W' :
         plt.title(r'$H\rightarrow WW$ Boosted Jet Image #'+str(i), fontsize = 22)
      if title == 'boost_HH4B' :
         plt.title(r'$H\rightarrow bb$ Boosted Jet Image #'+str(i), fontsize = 22)
      plt.xlabel(r'$\phi$', fontsize = 18)
      plt.ylabel(r'cos($\theta$)', fontsize = 20)
      if plotPNG == True :
         plt.savefig('plots/'+title+'_jetImageNum'+str(i)+'.png')
      if plotPDF == True :
         plt.savefig('plots/'+title+'_jetImageNum'+str(i)+'.pdf')
      plt.close()
