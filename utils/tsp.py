import numpy as np

def create_cities(n_cities, map_size=100, seed=42):
    """
    Generates a NumPy array of (n_cities, 2) shape representing
    the (x, y) coordinates for each city. [24]
    """
    np.random.seed(seed)
    cities = np.random.rand(n_cities, 2) * map_size
    return cities