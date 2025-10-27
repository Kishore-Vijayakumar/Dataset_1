# plant_data_loader.py
import pandas as pd
import numpy as np

def load_plant_village_data():
    """Simulate loading PlantVillage dataset"""
    # In actual implementation, load from Kaggle dataset
    data = {
        'image_path': ['tomato_healthy.jpg', 'tomato_bacterial_spot.jpg', 
                      'tomato_early_blight.jpg', 'tomato_late_blight.jpg'],
        'disease_class': ['Healthy', 'Bacterial_Spot', 'Early_Blight', 'Late_Blight'],
        'confidence': [0.95, 0.87, 0.92, 0.89],
        'leaf_condition': ['Good', 'Poor', 'Fair', 'Poor']
    }
    return pd.DataFrame(data)

def load_rice_leaf_data():
    """Simulate loading rice leaf disease data"""
    data = {
        'image_id': [1, 2, 3, 4, 5],
        'disease_type': ['Brown_Spot', 'Leaf_Blast', 'Healthy', 'Bacterial_Blight', 'Healthy'],
        'severity': [0.7, 0.8, 0.1, 0.9, 0.05],
        'treatment_required': [True, True, False, True, False]
    }
    return pd.DataFrame(data)