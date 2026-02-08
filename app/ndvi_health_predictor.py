import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import glob
import os
import rasterio

def generate_training_data(samples=300):
    np.random.seed(42)
    ndvi = np.random.rand(samples)
    soil_moisture = np.random.rand(samples)
    rainfall = np.random.rand(samples) * 100  # mm

    labels = []
    for i in range(samples):
        if ndvi[i] < 0.2 or soil_moisture[i] < 0.2:
            labels.append("Bad")
        elif 0.2 <= ndvi[i] < 0.5:
            labels.append("Moderate")
        else:
            labels.append("Good")
    
    df = pd.DataFrame({
        'ndvi': ndvi,
        'soil_moisture': soil_moisture,
        'rainfall': rainfall,
        'health': labels
    })
    return df

def train_model():
    df = generate_training_data()
    X = df[['ndvi', 'soil_moisture', 'rainfall']]
    y = df['health']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def get_suggestions(label):
    if label == "Bad":
        return [
            "⚠️ Increase irrigation immediately.",
            "🧪 Apply NPK fertilizer after soil testing.",
            "🐛 Check for pests or diseases."
        ]
    elif label == "Moderate":
        return [
            "💧 Slightly increase watering.",
            "🌿 Apply foliar nutrient spray.",
            "👀 Monitor environmental stress closely."
        ]
    elif label == "Good":
        return [
            "✅ Maintain current practices.",
            "🌾 Apply precision fertilizer.",
            "📅 Plan harvest based on NDVI trends."
        ]


def load_ndvi_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ndvi = img / 255.0  
    return ndvi
def convert_tif_to_png(ndvi_tif):
    ndvi_png = os.path.splitext(ndvi_tif)[0] + ".png"
    with rasterio.open(ndvi_tif) as src:
        ndvi_data = src.read(1).astype(np.float32)
        ndvi_normalized = ((ndvi_data - np.nanmin(ndvi_data)) /
                           (np.nanmax(ndvi_data) - np.nanmin(ndvi_data)) * 255).astype(np.uint8)
    plt.imsave(ndvi_png, ndvi_normalized, cmap="RdYlGn")
    print(f"🖼️ Converted '{ndvi_tif}' to PNG: {ndvi_png}")
    return ndvi_png
def analyze_zones(ndvi_array, model, zone_size=10):
    h, w = ndvi_array.shape
    results = []

    for y in range(0, h, zone_size):
        for x in range(0, w, zone_size):
            zone = ndvi_array[y:y+zone_size, x:x+zone_size]
            avg_ndvi = np.mean(zone)
            soil_moisture = np.random.uniform(0.3, 0.7) 
            rainfall = np.random.uniform(20, 100)       

            features = pd.DataFrame([{
                'ndvi': avg_ndvi,
                'soil_moisture': soil_moisture,
                'rainfall': rainfall
            }])

            prediction = model.predict(features)[0]
            suggestions = get_suggestions(prediction)

            results.append({
                'zone': f'({x},{y})',
                'avg_ndvi': round(avg_ndvi, 3),
                'soil_moisture': round(soil_moisture, 2),
                'rainfall': round(rainfall, 1),
                'health': prediction,
                'suggestions': suggestions
            })

    return results

def run_pipeline(image_path):
    print("📥 Loading image and training model...")
    model = train_model()
    ndvi = load_ndvi_image(image_path)
    results = analyze_zones(ndvi, model)

    print("\n📊 Zone-wise Crop Health & Suggestions:\n")
    for r in results:
        print(f"📍 Zone {r['zone']} | NDVI: {r['avg_ndvi']} | Health: {r['health']}")
        for s in r['suggestions']:
            print(f"   - {s}")
        print()

    plt.imshow(ndvi, cmap='YlGn')
    plt.title("NDVI Map")
    plt.colorbar(label="NDVI")
    plt.show()

if __name__ == "__main__":
    tif_files = sorted(glob.glob("ndvi_*.tif"), reverse=True)
    if not tif_files:
        raise FileNotFoundError("❌ No NDVI .tif files found in the current directory.")

    latest_tif = tif_files[0]
    image_path = convert_tif_to_png(latest_tif) 
    run_pipeline(image_path)
