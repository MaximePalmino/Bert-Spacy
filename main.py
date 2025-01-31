import numpy as np
import pandas as pd
import spacy
import requests
from geopy.distance import geodesic
from transformers import pipeline


class FrenchTrainRouteNLP:
    def __init__(self):
        self.nlp = spacy.load('fr_core_news_lg')
        self.train_stations = self._load_train_stations()
        self.city_coordinates = self._load_city_coordinates()
        self.intent_classifier = pipeline('text-classification',
                                          model='xlm-roberta-base')

    def _load_train_stations(self):
        stations_data = [
            {'name': 'Belfort TGV', 'lat': 47.6333, 'lon': 6.8667, 'city': 'Belfort'},
            {'name': 'Paris Gare de Lyon', 'lat': 48.8442, 'lon': 2.3744, 'city': 'Paris'},
            {'name': 'Lyon Part-Dieu', 'lat': 45.7609, 'lon': 4.8580, 'city': 'Lyon'},
            {'name': 'Marseille Saint-Charles', 'lat': 43.3023, 'lon': 5.3797, 'city': 'Marseille'},
            {'name': 'Toulouse Matabiau', 'lat': 43.6101, 'lon': 1.4531, 'city': 'Toulouse'},
            {'name': 'Nice-Ville', 'lat': 43.7031, 'lon': 7.2619, 'city': 'Nice'},
            {'name': 'Nantes', 'lat': 47.2173, 'lon': -1.5569, 'city': 'Nantes'},
            {'name': 'Strasbourg', 'lat': 48.5846, 'lon': 7.7366, 'city': 'Strasbourg'},
            {'name': 'Montpellier Saint-Roch', 'lat': 43.6045, 'lon': 3.8791, 'city': 'Montpellier'},
            {'name': 'Bordeaux Saint-Jean', 'lat': 44.8255, 'lon': -0.5587, 'city': 'Bordeaux'},
            {'name': 'Lille Flandres', 'lat': 50.6368, 'lon': 3.0700, 'city': 'Lille'},
            {'name': 'Rennes', 'lat': 48.1032, 'lon': -1.6720, 'city': 'Rennes'},
            {'name': 'Reims', 'lat': 49.2599, 'lon': 4.0254, 'city': 'Reims'},
            {'name': 'Le Havre', 'lat': 49.4942, 'lon': 0.1079, 'city': 'Le Havre'},
            {'name': 'Saint-Étienne Châteaucreux', 'lat': 45.4438, 'lon': 4.4036, 'city': 'Saint-Étienne'},
            {'name': 'Toulon', 'lat': 43.1257, 'lon': 5.9305, 'city': 'Toulon'},
            {'name': 'Grenoble', 'lat': 45.1906, 'lon': 5.7134, 'city': 'Grenoble'},
            {'name': 'Dijon Ville', 'lat': 47.3216, 'lon': 5.0368, 'city': 'Dijon'},
            {'name': 'Angers Saint-Laud', 'lat': 47.4675, 'lon': -0.5562, 'city': 'Angers'},
            {'name': 'Nîmes', 'lat': 43.8345, 'lon': 4.3611, 'city': 'Nîmes'},
            {'name': 'Aix-en-Provence TGV', 'lat': 43.4559, 'lon': 5.3179, 'city': 'Aix-en-Provence'},
            {'name': 'Brest', 'lat': 48.3885, 'lon': -4.5000, 'city': 'Brest'},
            {'name': 'Limoges Bénédictins', 'lat': 45.8345, 'lon': 1.2612, 'city': 'Limoges'},
            {'name': 'Tours', 'lat': 47.3919, 'lon': 0.6892, 'city': 'Tours'},
            {'name': 'Clermont-Ferrand', 'lat': 45.7792, 'lon': 3.0828, 'city': 'Clermont-Ferrand'},
            {'name': 'Amiens', 'lat': 49.8957, 'lon': 2.2979, 'city': 'Amiens'},
            {'name': 'Perpignan', 'lat': 42.6985, 'lon': 2.8945, 'city': 'Perpignan'},
            {'name': 'Metz-Ville', 'lat': 49.1090, 'lon': 6.1778, 'city': 'Metz'},
            {'name': 'Besançon Viotte', 'lat': 47.2445, 'lon': 6.0215, 'city': 'Besançon'},
            {'name': 'Orléans', 'lat': 47.9033, 'lon': 1.9016, 'city': 'Orléans'},
            {'name': 'Rouen Rive Droite', 'lat': 49.4474, 'lon': 1.0931, 'city': 'Rouen'},
            {'name': 'Mulhouse Ville', 'lat': 47.7431, 'lon': 7.3411, 'city': 'Mulhouse'},
            {'name': 'Caen', 'lat': 49.1844, 'lon': -0.3565, 'city': 'Caen'},
            {'name': 'Nancy', 'lat': 48.6838, 'lon': 6.1723, 'city': 'Nancy'},
            {'name': 'Saint-Denis', 'lat': 48.9356, 'lon': 2.3534, 'city': 'Saint-Denis'},
            {'name': 'Calais Ville', 'lat': 50.9581, 'lon': 1.8525, 'city': 'Calais'},
            {'name': 'Colmar', 'lat': 48.0793, 'lon': 7.3558, 'city': 'Colmar'},
            {'name': 'La Rochelle', 'lat': 46.1581, 'lon': -1.1446, 'city': 'La Rochelle'},
            {'name': 'Annecy', 'lat': 45.8991, 'lon': 6.1297, 'city': 'Annecy'},
            {'name': 'Pau', 'lat': 43.3013, 'lon': -0.3707, 'city': 'Pau'},
            {'name': 'Valence TGV', 'lat': 44.9742, 'lon': 4.9758, 'city': 'Valence'},
            {'name': 'Vannes', 'lat': 47.6559, 'lon': -2.7606, 'city': 'Vannes'},
            {'name': 'Quimper', 'lat': 47.9959, 'lon': -4.0977, 'city': 'Quimper'},
            {'name': 'Ajaccio', 'lat': 41.9184, 'lon': 8.7381, 'city': 'Ajaccio'},
        ]

        return pd.DataFrame(stations_data)

    def _load_city_coordinates(self):
        """
        Load French city coordinates
        """
        # TODO: Populate with comprehensive French city coordinate data
        city_data = [
            {'name': 'Belfort', 'lat': 47.6333, 'lon': 6.8667},
            {'name': 'Paris', 'lat': 48.8566, 'lon': 2.3522},
            {'name': 'Lyon', 'lat': 45.7640, 'lon': 4.8357},
            {'name': 'Marseille', 'lat': 43.2965, 'lon': 5.3698},
            {'name': 'Toulouse', 'lat': 43.6047, 'lon': 1.4442},
            {'name': 'Nice', 'lat': 43.7102, 'lon': 7.2620},
            {'name': 'Nantes', 'lat': 47.2184, 'lon': -1.5536},
            {'name': 'Strasbourg', 'lat': 48.5734, 'lon': 7.7521},
            {'name': 'Montpellier', 'lat': 43.6108, 'lon': 3.8767},
            {'name': 'Bordeaux', 'lat': 44.8378, 'lon': -0.5792},
            {'name': 'Lille', 'lat': 50.6292, 'lon': 3.0573},
            {'name': 'Rennes', 'lat': 48.1173, 'lon': -1.6778},
            {'name': 'Reims', 'lat': 49.2583, 'lon': 4.0317},
            {'name': 'Le Havre', 'lat': 49.4944, 'lon': 0.1079},
            {'name': 'Saint-Étienne', 'lat': 45.4397, 'lon': 4.3872},
            {'name': 'Toulon', 'lat': 43.1242, 'lon': 5.9280},
            {'name': 'Grenoble', 'lat': 45.1885, 'lon': 5.7245},
            {'name': 'Dijon', 'lat': 47.3220, 'lon': 5.0415},
            {'name': 'Angers', 'lat': 47.4784, 'lon': -0.5632},
            {'name': 'Nîmes', 'lat': 43.8367, 'lon': 4.3601},
            {'name': 'Aix-en-Provence', 'lat': 43.5297, 'lon': 5.4474},
            {'name': 'Brest', 'lat': 48.3904, 'lon': -4.4861},
            {'name': 'Limoges', 'lat': 45.8336, 'lon': 1.2611},
            {'name': 'Tours', 'lat': 47.3941, 'lon': 0.6848},
            {'name': 'Clermont-Ferrand', 'lat': 45.7797, 'lon': 3.0863},
            {'name': 'Amiens', 'lat': 49.8941, 'lon': 2.2957},
            {'name': 'Perpignan', 'lat': 42.6887, 'lon': 2.8948},
            {'name': 'Metz', 'lat': 49.1193, 'lon': 6.1757},
            {'name': 'Besançon', 'lat': 47.2378, 'lon': 6.0241},
            {'name': 'Orléans', 'lat': 47.9029, 'lon': 1.9092},
            {'name': 'Rouen', 'lat': 49.4432, 'lon': 1.0993},
            {'name': 'Mulhouse', 'lat': 47.7508, 'lon': 7.3359},
            {'name': 'Caen', 'lat': 49.1829, 'lon': -0.3707},
            {'name': 'Nancy', 'lat': 48.6921, 'lon': 6.1844},
            {'name': 'Saint-Denis', 'lat': 48.9356, 'lon': 2.3539},
            {'name': 'Argenteuil', 'lat': 48.9472, 'lon': 2.2467},
            {'name': 'Montreuil', 'lat': 48.8638, 'lon': 2.4485},
            {'name': 'Versailles', 'lat': 48.8014, 'lon': 2.1301},
            {'name': 'Boulogne-Billancourt', 'lat': 48.8356, 'lon': 2.2419},
            {'name': 'Avignon', 'lat': 43.9493, 'lon': 4.8055},
            {'name': 'Asnières-sur-Seine', 'lat': 48.9140, 'lon': 2.2873},
            {'name': 'Aubervilliers', 'lat': 48.9141, 'lon': 2.3841},
            {'name': 'Saint-Maur-des-Fossés', 'lat': 48.7899, 'lon': 2.4942},
            {'name': 'Calais', 'lat': 50.9513, 'lon': 1.8587},
            {'name': 'Bourges', 'lat': 47.0810, 'lon': 2.3988},
            {'name': 'Colmar', 'lat': 48.0793, 'lon': 7.3585},
            {'name': 'La Rochelle', 'lat': 46.1603, 'lon': -1.1511},
            {'name': 'Annecy', 'lat': 45.8992, 'lon': 6.1294},
            {'name': 'Bayonne', 'lat': 43.4929, 'lon': -1.4748},
            {'name': 'Niort', 'lat': 46.3237, 'lon': -0.4588},
            {'name': 'Chambéry', 'lat': 45.5646, 'lon': 5.9178},
            {'name': 'Poitiers', 'lat': 46.5802, 'lon': 0.3404},
            {'name': 'Pau', 'lat': 43.2951, 'lon': -0.3708},
            {'name': 'Angoulême', 'lat': 45.6484, 'lon': 0.1560},
            {'name': 'Valence', 'lat': 44.9334, 'lon': 4.8924},
            {'name': 'Tarbes', 'lat': 43.2321, 'lon': 0.0714},
            {'name': 'Lorient', 'lat': 47.7485, 'lon': -3.3669},
            {'name': 'Cholet', 'lat': 47.0666, 'lon': -0.8789},
            {'name': 'Albi', 'lat': 43.9283, 'lon': 2.1481},
            {'name': 'Évreux', 'lat': 49.0241, 'lon': 1.1510},
            {'name': 'Blois', 'lat': 47.5861, 'lon': 1.3359},
            {'name': 'Vannes', 'lat': 47.6582, 'lon': -2.7608},
            {'name': 'La Roche-sur-Yon', 'lat': 46.6705, 'lon': -1.4260},
            {'name': 'Cannes', 'lat': 43.5528, 'lon': 7.0174},
            {'name': 'Antibes', 'lat': 43.5804, 'lon': 7.1251},
            {'name': 'Beauvais', 'lat': 49.4293, 'lon': 2.0850},
            {'name': 'Saint-Nazaire', 'lat': 47.2806, 'lon': -2.2081},
            {'name': 'Meaux', 'lat': 48.9603, 'lon': 2.8887},
            {'name': 'Le Mans', 'lat': 48.0061, 'lon': 0.1996},
            {'name': 'Chartres', 'lat': 48.4522, 'lon': 1.4895},
            {'name': 'Quimper', 'lat': 47.9957, 'lon': -4.0970},
            {'name': 'Gap', 'lat': 44.5594, 'lon': 6.0786},
            {'name': 'Ajaccio', 'lat': 41.9192, 'lon': 8.7386},
            {'name': 'Bastia', 'lat': 42.7029, 'lon': 9.4508},
            {'name': 'Alès', 'lat': 44.1255, 'lon': 4.0837},
            {'name': 'Castres', 'lat': 43.6048, 'lon': 2.240}
        ]
        return pd.DataFrame(city_data)

    def find_closest_station(self, location):
        location = self._normalize_location(location)

        station_match = self.train_stations[
            self.train_stations['name'].str.contains(location, case=False)
        ]
        if not station_match.empty:
            return station_match.iloc[0]

        city_coords = self._find_city_coordinates(location)

        stations_with_distance = self.train_stations.copy()
        stations_with_distance['distance'] = stations_with_distance.apply(
            lambda row: geodesic(
                (city_coords['lat'], city_coords['lon']),
                (row['lat'], row['lon'])
            ).kilometers,
            axis=1
        )

        return stations_with_distance.sort_values('distance').iloc[0]

    def _normalize_location(self, location):
        doc = self.nlp(location)
        loc_entities = [ent.text for ent in doc.ents if ent.label_ == 'LOC']
        return loc_entities[0] if loc_entities else location

    def _find_city_coordinates(self, city_name):
        city_match = self.city_coordinates[
            self.city_coordinates['name'].str.contains(city_name, case=False)
        ]
        if city_match.empty:
            raise ValueError(f"Could not find coordinates for {city_name}")
        return city_match.iloc[0]

    def get_train_route(self, start_location, end_location):
        start_station = self.find_closest_station(start_location)
        end_station = self.find_closest_station(end_location)

        # TODO: Integrate with actual train routing API
        return {
            'start_station': start_station['name'],
            'end_station': end_station['name'],
            'distance': geodesic(
                (start_station['lat'], start_station['lon']),
                (end_station['lat'], end_station['lon'])
            ).kilometers
        }


nlp = FrenchTrainRouteNLP()
route = nlp.get_train_route("Paris", "Belfort")
print(route)