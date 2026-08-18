import folium
from folium.plugins import Draw

# Create an interactive map with satellite imagery + labels (Google Hybrid)
m = folium.Map(
    location=[30.0, 70.0],
    zoom_start=5,
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google",
    max_zoom=20
)

# Add drawing tools
draw = Draw(export=True, filename="aoi.geojson")
draw.add_to(m)

# Save map as an HTML file
m.save("interactive_map.html")

print("Open 'interactive_map.html' in a browser, draw AOI, and download 'aoi.geojson'.")
