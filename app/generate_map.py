import folium
from folium.plugins import Draw

# Create an interactive map
m = folium.Map(location=[30.0, 70.0], zoom_start=5)

# Add drawing tools
draw = Draw(export=True, filename="aoi.geojson")
draw.add_to(m)

# Save map as an HTML file
m.save("interactive_map.html")

print("Open 'interactive_map.html' in a browser, draw AOI, and download 'aoi.geojson'.")
