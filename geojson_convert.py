root, ext = os.path.splitext(__file__)
mapfile = root  + '.html'
# Create the map. Save the file to basic_plot.html. _map.html is the default
# if 'path' is not specified
# mplleaflet.show(path=mapfile)

# Convert matplotlib contourf to geojson
geojson = geojsoncontour.contourf_to_geojson(
    contourf=contourf,
    min_angle_deg=3.0,
    ndigits=5,
    stroke_width=1,
    fill_opacity=0.5)

# Set up the folium plot
geomap = folium.Map([latCenter, lonCenter], zoom_start=10, tiles="cartodbpositron")

# Plot the contour plot on folium
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        'color':     x['properties']['stroke'],
        'weight':    x['properties']['stroke-width'],
        'fillColor': x['properties']['fill'],
        'opacity':   0.6,
    }).add_to(geomap)

# Plot the data
geomap.save(mapfile)
