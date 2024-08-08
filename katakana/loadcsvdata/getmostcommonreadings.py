from collections import Counter

def get_most_common_readings(data):
    """ Function to get the most common Katakana reading for each English word """
    print("Counting frequency of Katakana readings for each English word...")
    most_common_readings = data.groupby('english')['katakana'].apply(lambda x: Counter(x).most_common(1)[0][0]).reset_index()
    return most_common_readings
