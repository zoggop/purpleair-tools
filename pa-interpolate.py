from purpleair.network import SensorList
from purpleair.sensor import Sensor
import matplotlib.pyplot as plt
# import mplleaflet
import numpy as np
from scipy.interpolate import griddata
from scipy import ndimage
import sys
import os
# import folium
# import geojsoncontour
# import pandas as pd
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import io
from urllib.request import urlopen, Request
from PIL import Image
import pickle
import datetime
import subprocess
import re
from colormath.color_objects import sRGBColor, LCHabColor, LabColor
from colormath.color_conversions import convert_color
import concurrent.futures
import threading
import time

# station IDs to exclude
excludeList = [66405, 35379]

excludedIDs = {}
for sid in excludeList:
	excludedIDs[sid] = True

lonBounds = [-123.72, -122.16] # [-124.67, -121.65] # [-125, -102]
latBounds = [42.01, 42.61] # [41.00, 42.99] #[29, 50]

# data coordinates and values
x = []
y = []
z = []

Is = 			[[0,50], 	[51,100], 		[101,150], 							[151,200], 		[201,300], 			[301,400], [401,500]]
BPs = 			[[0,12], 	[12.1,35.4], 	[35.5,55.4], 						[55.5,150.4], 	[150.5,250.4], 		[250.5,350.4], [350.5,500.4]]
categories = 	['Good', 	'Moderate', 	'Unhealthy for Sensitive Groups', 	'Unhealthy', 	'Very Unhealthy', 	'Hazardous', 'Hazardous']
aqiSlopes = []
for n in range(0, len(BPs)):
		BPwith = BPs[n]
		Iwith = Is[n]
		slope = (Iwith[1] - Iwith[0]) / (BPwith[1] - BPwith[0])
		aqiSlopes.append(slope)

# conBreaks = [0, 12, 35.4, 55.4, 150.4, 250.4]
conBreaks = [0, 23.7, 45.4, 102.9, 200.4, 300.4]
breakHexes = ['#07e600', '#fbff00', '#fa8003', '#fa040b', '#903a99', '#7c001f']
breaksRGBs = []
breaksLCHs = []
breaksLABs = []
breakColors = []
for hexColor in breakHexes:
	srgb = sRGBColor.new_from_rgb_hex(hexColor)
	lch = convert_color(srgb, LCHabColor)
	lab = convert_color(srgb, LabColor)
	breaksRGBs.append(srgb)
	breaksLCHs.append(lch)
	breaksLABs.append(lab)
	breakColors.append(list(srgb.get_value_tuple()))
	# print(hexColor, srgb, lch, list(srgb.get_value_tuple()))

def gradientColor(con, model):
	index = None
	for i in range(0, len(conBreaks)-1):
		if con == conBreaks[i]:
			return breakColors[i]
		elif con > conBreaks[i] and con < conBreaks[i+1]:
			index = i
			break
	if index == None:
		return breakColors[-1]
	conAboveA = (con - conBreaks[index]) / (conBreaks[index+1] - conBreaks[index])
	if model == 'lch':
		a = breaksLCHs[index]
		b = breaksLCHs[index+1]
		l = a.lch_l + ((b.lch_l - a.lch_l) * conAboveA)
		c = a.lch_c + ((b.lch_c - a.lch_c) * conAboveA)
		if abs(b.lch_h - a.lch_h) > 180:
			if b.lch_h > a.lch_h:
				hDist = (b.lch_h - 360) - a.lch_h
			else:
				hDist = (b.lch_h + 360) - a.lch_h
			h = a.lch_h + (hDist * conAboveA)
			if h > 360:
				h = h - 360
			if h < 0:
				h = h + 360
		else:
			h = a.lch_h + ((b.lch_h - a.lch_h) * conAboveA)
		srgb = convert_color(LCHabColor(l, c, h), sRGBColor)
	elif model == 'rgb':
		a = breaksRGBs[index]
		b = breaksRGBs[index+1]
		red = a.rgb_r + ((b.rgb_r - a.rgb_r) * conAboveA)
		green = a.rgb_g + ((b.rgb_g - a.rgb_g) * conAboveA)
		blue = a.rgb_b + ((b.rgb_b - a.rgb_b) * conAboveA)
		srgb = sRGBColor(red, green, blue)
	elif model == 'lab':
		first = breaksLABs[index]
		second = breaksLABs[index+1]
		l = first.lab_l + ((second.lab_l - first.lab_l) * conAboveA)
		a = first.lab_a + ((second.lab_a - first.lab_a) * conAboveA)
		b = first.lab_b + ((second.lab_b - first.lab_b) * conAboveA)
		srgb = convert_color(LabColor(l, a, b), sRGBColor)
	# return [int(srgb.clamped_rgb_r * 255), int(srgb.clamped_rgb_g * 255), int(srgb.clamped_rgb_b * 255)]
	return [srgb.clamped_rgb_r, srgb.clamped_rgb_g, srgb.clamped_rgb_b]

startDT = datetime.datetime.now()
gradient = {}
for con in np.arange(conBreaks[0], conBreaks[-1], 0.1):
	con = round(con,1)
	gradient[con] = gradientColor(con, 'lab')
	# print(con, gradient[con])
	# gradient.append(gradientColor(con, 'rgb'))

def colorFromGradient(con):
	if np.isnan(con):
		# print("NaN color")
		return gradient.get(conBreaks[0])
	else:
		color = gradient.get(round(con,1))
		if color == None:
			# print(con, round(con,1))
			return gradient.get(conBreaks[-1])
		else:
			return color

def getMyLocation():
	import geocoder
	args = None
	if len(sys.argv) > 1:
		args = ' '.join(sys.argv[1:])
		if not args == None:
			# get location of argument placename through OSM
			g = geocoder.osm(args)
			return g.latlng[0], g.latlng[1]
	# try getting it with a script
	result = subprocess.run([os.path.expanduser('~/scripts/location.bat')], stdout=subprocess.PIPE)
	if not result == None:
		pattern = re.compile(r'(-?[\d.]+)')
		matches = pattern.findall(result.stdout.decode('utf-8'))
		if matches != None and len(matches) > 0:
			print("got location from script")
			return float(matches[0]), float(matches[1])
	else:
		# try getting location of IP through duckduckgo and OSM
		response = urlopen('https://duckduckgo.com/?q=my+ip&t=hx&va=g&ia=answer')
		pattern = re.compile(r'Your IP address is.+?\>([\w\s,]+)')
		matches = pattern.findall(response.read().decode('utf-8'))
		if matches != None and len(matches) > 0:
			print(matches[0])
		g = geocoder.osm(matches[0])
	return g.latlng[0], g.latlng[1]

def getRelevantSensors():
	# create path if not existant
	if not os.path.exists(os.path.expanduser('~/.purple_epa')):
		os.makedirs(os.path.expanduser('~/.purple_epa'))
	# load saved list if available
	if os.path.exists(os.path.expanduser('~/.purple_epa/relevantsensors.pickle')):
		mtime = os.path.getmtime(os.path.expanduser('~/.purple_epa/relevantsensors.pickle'))
		last_modified_date = datetime.datetime.fromtimestamp(mtime)
		listAge = datetime.datetime.now() - last_modified_date
		print('saved sensor list is', listAge.days, 'days old')
		if listAge.days < 1:
			with open(os.path.expanduser('~/.purple_epa/relevantsensors.pickle'), "rb") as fp:
				rs = pickle.load(fp)
			fp.close()
			return rs
	# create list
	rs = []
	p = SensorList()
	isOutsideParents = {}
	parents = {}
	dfP = p.to_dataframe(sensor_filter='useful', channel='parent')
	for index, row in dfP.iterrows():
		if row['location_type'] == 'outside':
			parents[index] = row
			isOutsideParents[index] = True
	dfC = p.to_dataframe(sensor_filter='useful', channel='child')
	for index, row in dfC.iterrows():
		if type(row['parent']) is float:
			parentIndex = int(row['parent'])
			isOutsideParent = isOutsideParents.get(parentIndex)
			if not isOutsideParent == None:
				outsideParent = parents.get(parentIndex)
				rs.append({'index': parentIndex, 'lat': outsideParent['lat'], 'lon': outsideParent['lon']})
	# save list
	with open(os.path.expanduser('~/.purple_epa/relevantsensors.pickle'), "wb") as fp:
		pickle.dump(rs, fp)
	fp.close()
	return rs

def epaCorrect(s):
	if s.parent.channel_data.get('pm2_5_cf_1') == None or s.child.channel_data.get('pm2_5_cf_1') == None:
		return None
	if abs(s.parent.d1avg - s.child.d1avg) > 5:
		# channels must agree
		print(s.parent.identifier, s.parent.name)
		return None
	a = float(s.parent.channel_data.get('pm2_5_cf_1'))
	b = float(s.child.channel_data.get('pm2_5_cf_1'))
	avgAB = (a + b) / 2
	humidity = s.parent.current_humidity # (s.parent.current_humidity + s.child.current_humidity) / 2
	temp = s.parent.current_temp_f
	pressure = s.parent.current_pressure
	epa = (0.534 * avgAB) - (0.00844 * humidity) + 5.6044
	return epa

def aqiFromConcentration(epa):
	category = None
	BPwith = None
	Iwith = None
	slope = None
	for n in range(0, len(BPs)):
		BP = BPs[n]
		if epa >= BP[0]-0.1 and epa <= BP[1]:
			BPwith = BP
			Iwith = Is[n]
			slope = aqiSlopes[n]
			category = categories[n]
			break
	if category == None:
		slope = 1
		BPwith = BPs[-1]
		Iwith = Is[-1]
	aqi = (slope * (epa - BPwith[0])) + Iwith[0]
	if np.isnan(aqi):
		aqi = 0
		# print('NaN aqi', epa)
	return aqi

aqiByCon = {}
for con in np.arange(0, BPs[-1][1], 0.1):
	con = round(con, 1)
	aqiByCon[con] = int(aqiFromConcentration(con))

def aqiFast(epa):
	return aqiByCon.get(round(epa,1)) or aqiFromConcentration(epa)

def pollSensor(entry):
	sid = entry['index']
	entry['sensor'] = Sensor(sid)

def pollAllSensors(entries):
	startDT = datetime.datetime.now()
	with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
		executor.map(pollSensor, entries)
	print("downloaded individual sensor data in", datetime.datetime.now() - startDT)


centerLatLon = getMyLocation()
lonBounds = [centerLatLon[1] - 1, centerLatLon[1] + 1]
latBounds = [centerLatLon[0] - 0.4, centerLatLon[0] + 0.4]

lonBoundsRange = lonBounds[1] - lonBounds[0]
latBoundsRange = latBounds[1] - latBounds[0]
sensorLonBounds = [lonBounds[0] - (lonBoundsRange / 1.5), lonBounds[1] + (lonBoundsRange / 1.5)]
sensorLatBounds = [latBounds[0] - (latBoundsRange / 1.5), latBounds[1] + (latBoundsRange / 1.5)]
print(centerLatLon, latBounds, lonBounds, sensorLatBounds, sensorLonBounds)

relevant = getRelevantSensors()
# Other sensor filters include 'outside', 'useful', 'family', and 'no_child'
zMin, zMax = None, None
nowDT = datetime.datetime.utcnow()
pollList = []
for entry in relevant:
	if not excludedIDs.get(entry['index']):
		lon = entry['lon']
		lat = entry['lat']
		if lon > sensorLonBounds[0] and lon < sensorLonBounds[1] and lat > sensorLatBounds[0] and lat < sensorLatBounds[1]:
			sid = entry['index']
			pollList.append(entry)
pollAllSensors(pollList)
for entry in pollList:
	lon = entry['lon']
	lat = entry['lat']
	s = entry['sensor']
	# seen = s.parent.last_seen.tz_localize('UTC')
	age = nowDT - s.parent.last_seen
	# print(age, age.seconds)
	if age.days == 0 and age.seconds < 3600:
		con = epaCorrect(s)
		if con:
			x.append(lon)
			y.append(lat)
			z.append(con)
			if zMin == None or con < zMin:
				zMin = con
			if zMax == None or con > zMax:
				zMax = con
zRange = zMax - zMin
print(len(z))
print(zMin, zMax)

# target grid to interpolate to
xi = np.arange(lonBounds[0],lonBounds[1],0.002)
yi = np.arange(latBounds[0],latBounds[1],0.002)
xi,yi = np.meshgrid(xi,yi)

# interpolate
zi = griddata((x,y),z,(xi,yi),method='linear')
# blur
zi = ndimage.gaussian_filter(zi, sigma=3)

# init rgb image of data
# rgbMap = np.zeros([zi.shape[0], zi.shape[1], 3], dtype=np.uint8)
rgbMap = np.zeros([zi.shape[0], zi.shape[1], 3])
# color interpolated values according to gradient
for i, con in np.ndenumerate(zi):
	rgb = colorFromGradient(con)
	rgbMap[i] = rgb
	if int(aqiFast(con)) == 125:
		if rgb[2] > rgb[1]:
			print(con, round(con,1), rgb)

# create map of aqi values
aqiMap = np.zeros([zi.shape[0], zi.shape[1]])
for i, con in np.ndenumerate(zi):
	aqi = aqiFast(con)
	aqiMap[i] = int(aqi)

def image_spoof(self, tile): # this function pretends not to be a Python script
	url = self._image_url(tile) # get the url of the street map API
	req = Request(url) # start request
	req.add_header('User-agent','Anaconda 3') # add user agent to request
	# req.add_header('Desired-tile-form', 'BW')
	fh = urlopen(req) 
	im_data = io.BytesIO(fh.read()) # get image
	fh.close() # close url
	img = Image.open(im_data) # open image with PIL
	img = img.convert(self.desired_tile_form) # set image format
	return img, self.tileextent(tile), 'lower' # reformat for cartopy

cimgt.OSM.get_image = image_spoof # reformat web request for street map spoofing
osm_img = cimgt.OSM(desired_tile_form='L') # spoofed, downloaded street map

fig = plt.figure(figsize=(12,9)) # open matplotlib figure
ax1 = plt.axes(projection=osm_img.crs) # project using coordinate reference system (CRS) of street map
# center_pt = [latCenter, lonCenter] # lat/lon of One World Trade Center in NYC
# zoom = 0.00075 # for zooming out of center point
# extent = [center_pt[1]-(zoom*2.0),center_pt[1]+(zoom*2.0),center_pt[0]-zoom,center_pt[0]+zoom] # adjust to zoom
extent = [lonBounds[0], lonBounds[1], latBounds[0], latBounds[1]]
ax1.set_extent(extent) # set extents
ax1.add_image(osm_img, 10, cmap='gray') # add OSM with zoom specification
# NOTE: zoom specifications should be selected based on extent:
# -- 2     = coarse image, select for worldwide or continental scales
# -- 4-6   = medium coarseness, select for countries and larger states
# -- 6-10  = medium fineness, select for smaller states, regions, and cities
# -- 10-12 = fine image, select for city boundaries and zip codes
# -- 14+   = extremely fine image, select for roads, blocks, buildings

# ax1.imshow(zi, origin='lower', alpha=0.5, extent=[lonBounds[0], lonBounds[1], latBounds[0], latBounds[1]], vmin=0, vmax=50, transform=ccrs.PlateCarree(), zorder=2, cmap='RdYlGn_r')
ax1.imshow(rgbMap, origin='lower', alpha=0.5, extent=[lonBounds[0], lonBounds[1], latBounds[0], latBounds[1]], transform=ccrs.PlateCarree(), zorder=2)
# show boundaries of AQI categories
ax1.contour(xi,yi,zi, [0, 12, 35.4, 55.4, 150.4, 250.4, 350.4], colors=('#07e600', '#fbff00', '#fa8003', '#fa040b', '#903a99', '#7c001f'), alpha=1, transform=ccrs.PlateCarree(), linestyles='dotted')
# transparent overlay of aqi values so that the smoke concentration at the cursor will be shown
ax1.imshow(aqiMap, origin='lower', alpha=0, extent=[lonBounds[0], lonBounds[1], latBounds[0], latBounds[1]], transform=ccrs.PlateCarree(), zorder=2)
# ax1.contourf(xi,yi,zi, [0, 12, 35, 55, 150, 250, 350], colors=('#07e600', '#fbff00', '#fa8003', '#fa040b', '#903a99', '#7c001f'), alpha=0.5, transform=ccrs.PlateCarree(), extend='both')
# ax1.scatter(x,y, transform=ccrs.PlateCarree())
# for i in range(0, len(z)):
	# ax1.annotate(str(int(z[i])), (x[i], y[i]), transform=ccrs.PlateCarree())
	# ax1.text(x[i], y[i], str(int(z[i])), transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show() # show the plot

exit()