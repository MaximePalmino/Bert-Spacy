import numpy as np
import pandas as pd
import requests
import logging
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FrenchTrainRouteNLP:
    def __init__(self, data_file='liste-des-gares.xlsx'):
        logger.info(f"Initializing FrenchTrainRouteNLP with data file: {data_file}")
        self.data_file = data_file

        logger.info("Loading NER model")
        try:
            self.ner_model = pipeline("ner", model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NER model: {str(e)}")
            raise

        logger.info("Loading train station data")
        try:
            self.train_stations = self._load_train_stations()
            logger.info(f"Loaded {len(self.train_stations)} train stations successfully")
        except Exception as e:
            logger.error(f"Failed to load train station data: {str(e)}")
            raise

        self.geolocator = Nominatim(user_agent="french_train_route")
        logger.info("Geocoder initialized successfully")

    def _load_train_stations(self):
        try:
            if self.data_file.endswith('.xlsx'):
                df = pd.read_excel(self.data_file)
            else:
                df = pd.read_csv(self.data_file, encoding='utf-8')

            stations = pd.DataFrame({
                'name': df['LIBELLE'],
                'code_uic': df['CODE_UIC'],
                'commune': df['COMMUNE'],
                'departement': df['DEPARTEMEN'],
                'lat': df['Y_WGS84'],
                'lon': df['X_WGS84'],
                'is_freight': df['FRET'] == 'O',
                'is_passenger': df['VOYAGEURS'] == 'O',
                'pk': df['PK'],
                'line_code': df['CODE_LIGNE']
            })

            stations['lat'] = pd.to_numeric(stations['lat'], errors='coerce')
            stations['lon'] = pd.to_numeric(stations['lon'], errors='coerce')

            stations = stations.dropna(subset=['lat', 'lon'])

            stations = stations[stations.apply(
                lambda row: self._is_valid_french_coordinates(row['lat'], row['lon']),
                axis=1
            )]

            if stations.empty:
                raise Exception("No valid station data was loaded")

            return stations

        except Exception as e:
            logger.error(f"Error loading station data: {str(e)}")
            raise

    def _is_valid_french_coordinates(self, lat, lon):
        return (41.0 <= lat <= 51.5) and (-5.0 <= lon <= 10.0)

    def _extract_locations(self, text):
        try:
            ner_results = self.ner_model(text)
            locations = [res["word"] for res in ner_results if res["entity_group"] == "LOC"]

            if len(locations) < 2:
                raise ValueError("Could not determine enough locations for a valid route.")

            logger.info(f"Filtered locations: {locations}")
            return locations  # Return all locations in the order they appear

        except Exception as e:
            logger.warning(f"NER processing failed: {str(e)}")
            raise ValueError("Could not extract locations from input text.")

    def _is_real_city(self, city_name):
        return city_name.lower() in self.train_stations['commune'].str.lower().unique()

    def _get_location_coordinates(self, location):
        try:
            logger.info(f"Geocoding location: {location}")
            location_data = self.geolocator.geocode(f"{location}, France")

            if location_data:
                coords = {'lat': location_data.latitude, 'lon': location_data.longitude}
                if self._is_valid_french_coordinates(coords['lat'], coords['lon']):
                    return coords
                else:
                    raise ValueError("Coordinates outside France")

            raise ValueError(f"Location not found: {location}")

        except Exception as e:
            logger.error(f"Geocoding error for {location}: {str(e)}")
            raise ValueError(f"Geocoding error: {str(e)}")

    def find_closest_station(self, location, passenger_only=True):

        stations = self.train_stations[self.train_stations['is_passenger']] if passenger_only else self.train_stations

        station_match = stations[(stations['name'].str.contains(location, case=False)) |
                                 (stations['commune'].str.contains(location, case=False))]

        if not station_match.empty:
            priority_stations = station_match[station_match['line_code'].astype(str).str.startswith('TGV')]
            if not priority_stations.empty:
                return priority_stations.iloc[0]
            return station_match.iloc[0]

        coords = self._get_location_coordinates(location)

        stations['distance'] = stations.apply(
            lambda row: geodesic((coords['lat'], coords['lon']), (row['lat'], row['lon'])).kilometers, axis=1
        )

        priority_stations = stations[stations['line_code'].str.startswith('TGV')]
        if not priority_stations.empty:
            closest_station = priority_stations.sort_values('distance').iloc[0]
        else:
            closest_station = stations.sort_values('distance').iloc[0]

        return closest_station

    def get_train_route(self, locations, passenger_only=True):

        if len(locations) < 2:
            raise ValueError("At least two locations are required to calculate a route.")

        segments = []
        total_distance = 0

        for i in range(len(locations) - 1):
            start_station = self.find_closest_station(locations[i], passenger_only)
            end_station = self.find_closest_station(locations[i + 1], passenger_only)

            segment_distance = geodesic((start_station['lat'], start_station['lon']),
                                        (end_station['lat'], end_station['lon'])).kilometers

            total_distance += segment_distance

            segments.append({
                'start_station': {'name': start_station['name'], 'commune': start_station['commune'],
                                  'code_uic': start_station['code_uic']},
                'end_station': {'name': end_station['name'], 'commune': end_station['commune'],
                                'code_uic': end_station['code_uic']},
                'distance_km': segment_distance,
                'line_codes': {'start': start_station['line_code'], 'end': end_station['line_code']}
            })

        return {
            'total_distance_km': total_distance,
            'segments': segments
        }

    def get_train_route_from_text(self, user_input, passenger_only=True):
        try:
            locations = self._extract_locations(user_input)
            return self.get_train_route(locations, passenger_only)
        except ValueError as e:
            logger.error(f"Failed to process input: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    train_nlp = FrenchTrainRouteNLP()
    user_input = "Je dois poser quelque chose à Aubin-St-Vaast entre Chambly et Poliénas"
    route_info = train_nlp.get_train_route_from_text(user_input)

    print(f"Total Distance: {route_info.get('total_distance_km', 'N/A')} km")
    for segment in route_info.get('segments', []):
        print(f"{segment['start_station']['name']} → {segment['end_station']['name']}: {segment['distance_km']} km")
