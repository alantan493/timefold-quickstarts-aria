from typing import Generator, TypeVar, Sequence
from datetime import date, datetime, time, timedelta
from enum import Enum
from random import Random
from dataclasses import dataclass

from .domain import *


FIRST_NAMES = ("Amy", "Beth", "Carl", "Dan", "Elsa", "Flo", "Gus", "Hugo", "Ivy", "Jay", "Ken", "Lin", "Max", "Nina", "Owen", "Priya", "Quinn", "Rita", "Sam", "Tina")
LAST_NAMES = ("Cole", "Fox", "Green", "Jones", "King", "Li", "Poe", "Rye", "Smith", "Watt", "Chan", "Lim", "Tan", "Wong", "Ng", "Lee", "Ong", "Teo", "Goh", "Koh")
SERVICE_DURATION_MINUTES = (10, 15, 20, 25, 30, 40)
MORNING_WINDOW_START = time(7, 0)  # Extended from 8:00 to 7:00
MORNING_WINDOW_END = time(14, 0)   # Extended from 12:00 to 14:00
AFTERNOON_WINDOW_START = time(12, 0)  # Earlier start from 13:00 to 12:00
AFTERNOON_WINDOW_END = time(20, 0)    # Extended from 18:00 to 20:00
EVENING_WINDOW_START = time(18, 30)
EVENING_WINDOW_END = time(22, 0)      # Extended from 21:00 to 22:00


@dataclass
class _DemoDataProperties:
    seed: int
    visit_count: int
    vehicle_count: int
    vehicle_start_time: time
    min_demand: int
    max_demand: int
    min_vehicle_capacity: int
    max_vehicle_capacity: int
    south_west_corner: Location
    north_east_corner: Location

    def __post_init__(self):
        if self.min_demand < 1:
            raise ValueError(f"minDemand ({self.min_demand}) must be greater than zero.")
        if self.max_demand < 1:
            raise ValueError(f"maxDemand ({self.max_demand}) must be greater than zero.")
        if self.min_demand >= self.max_demand:
            raise ValueError(f"maxDemand ({self.max_demand}) must be greater than minDemand ({self.min_demand}).")
        if self.min_vehicle_capacity < 1:
            raise ValueError(f"Number of minVehicleCapacity ({self.min_vehicle_capacity}) must be greater than zero.")
        if self.max_vehicle_capacity < 1:
            raise ValueError(f"Number of maxVehicleCapacity ({self.max_vehicle_capacity}) must be greater than zero.")
        if self.min_vehicle_capacity >= self.max_vehicle_capacity:
            raise ValueError(f"maxVehicleCapacity ({self.max_vehicle_capacity}) must be greater than "
                             f"minVehicleCapacity ({self.min_vehicle_capacity}).")
        if self.visit_count < 1:
            raise ValueError(f"Number of visitCount ({self.visit_count}) must be greater than zero.")
        if self.vehicle_count < 1:
            raise ValueError(f"Number of vehicleCount ({self.vehicle_count}) must be greater than zero.")
        if self.north_east_corner.latitude <= self.south_west_corner.latitude:
            raise ValueError(f"northEastCorner.getLatitude ({self.north_east_corner.latitude}) must be greater than "
                             f"southWestCorner.getLatitude({self.south_west_corner.latitude}).")
        if self.north_east_corner.longitude <= self.south_west_corner.longitude:
            raise ValueError(f"northEastCorner.getLongitude ({self.north_east_corner.longitude}) must be greater than "
                             f"southWestCorner.getLongitude({self.south_west_corner.longitude}).")


class DemoData(Enum):
    # SIMPLE TEST - 1 vehicle, 5 visits for hybrid testing
    SIMPLE = _DemoDataProperties(1000, 5, 1, time(8, 0),
                                1, 2, 20, 30,
                                Location(latitude=1.2700, longitude=103.8200),
                                Location(latitude=1.3000, longitude=103.8700))
    
    # SMALL TEST - 1 vehicle, 10 visits for brute force testing
    SMALL = _DemoDataProperties(1001, 10, 1, time(8, 0),
                               1, 2, 20, 30,
                               Location(latitude=1.2700, longitude=103.8200),
                               Location(latitude=1.3000, longitude=103.8700))

    # Central Singapore - Reduced for testing
    CENTRAL = _DemoDataProperties(2001, 15, 2, time(8, 0),  # REDUCED: 15 visits, 2 vehicles
                                  1, 2, 15, 30,
                                  Location(latitude=1.2700, longitude=103.8200),
                                  Location(latitude=1.3000, longitude=103.8700))

    # Singapore Island-Wide - Reduced for stability
    SINGAPORE_WIDE = _DemoDataProperties(3001, 30, 3, time(8, 0),  # REDUCED: 30 visits, 3 vehicles
                                         1, 4, 20, 40,
                                         Location(latitude=1.2400, longitude=103.6000),
                                         Location(latitude=1.4700, longitude=104.0200))

    # Large Scale Test - For future stress testing
    LARGE_SCALE = _DemoDataProperties(4001, 50, 5, time(8, 0),  # REDUCED: 50 visits, 5 vehicles
                                      1, 6, 25, 50,
                                      Location(latitude=1.2400, longitude=103.6000),
                                      Location(latitude=1.4700, longitude=104.0200))


def doubles(random: Random, start: float, end: float) -> Generator[float, None, None]:
    while True:
        yield random.uniform(start, end)


def ints(random: Random, start: int, end: int) -> Generator[int, None, None]:
    while True:
        yield random.randrange(start, end)


T = TypeVar('T')


def values(random: Random, sequence: Sequence[T]) -> Generator[T, None, None]:
    start = 0
    end = len(sequence) - 1
    while True:
        yield sequence[random.randint(start, end)]


def generate_names(random: Random) -> Generator[str, None, None]:
    while True:
        yield f'{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}'


def get_singapore_locations_for_region(demo_data_enum: DemoData, count: int):
    """Optimized location distribution with guaranteed east/west spread."""
    
    print(f"📍 Generating {count} locations for {demo_data_enum.name}")
    
    # Hardcoded strategic locations for maximum coverage
    
    # Central areas - Primary locations for testing
    central_locations = [
        Location(latitude=1.2966, longitude=103.8518),  # Marina Bay
        Location(latitude=1.2839, longitude=103.8519),  # Raffles Place
        Location(latitude=1.3048, longitude=103.8318),  # Orchard
        Location(latitude=1.2792, longitude=103.8480),  # Chinatown
        Location(latitude=1.3521, longitude=103.8198),  # City Hall
        Location(latitude=1.3000, longitude=103.8500),  # Bugis
        Location(latitude=1.2800, longitude=103.8400),  # Clarke Quay
        Location(latitude=1.2900, longitude=103.8600),  # Marina Centre
    ]
    
    # East areas
    east_locations = [
        Location(latitude=1.3644, longitude=103.9915),  # Changi Airport
        Location(latitude=1.3496, longitude=103.9568),  # Tampines
        Location(latitude=1.3329, longitude=103.9436),  # Bedok
        Location(latitude=1.3240, longitude=103.9520),  # Tanah Merah
    ]
    
    # West areas
    west_locations = [
        Location(latitude=1.3387, longitude=103.7053),  # Jurong East
        Location(latitude=1.3553, longitude=103.6874),  # Boon Lay
        Location(latitude=1.3477, longitude=103.7430),  # Jurong West
        Location(latitude=1.3200, longitude=103.6800),  # Pioneer
    ]
    
    # North areas
    north_locations = [
        Location(latitude=1.4382, longitude=103.7890),  # Woodlands
        Location(latitude=1.4294, longitude=103.8356),  # Yishun
        Location(latitude=1.4168, longitude=103.8432),  # Ang Mo Kio
        Location(latitude=1.3966, longitude=103.8467),  # Bishan
    ]
    
    locations = []
    
    if demo_data_enum in [DemoData.SIMPLE, DemoData.SMALL]:
        # Use only central locations for simple testing
        for i in range(count):
            locations.append(central_locations[i % len(central_locations)])
        print(f"📍 Distribution: {count} Central (simple test)")
            
    elif demo_data_enum == DemoData.CENTRAL:
        # Mostly central with some variety
        central_count = max(1, count * 3 // 4)  # 75% central
        other_count = count - central_count
        
        for i in range(central_count):
            locations.append(central_locations[i % len(central_locations)])
        
        # Add some variety from other areas
        other_locations = east_locations + west_locations
        for i in range(other_count):
            locations.append(other_locations[i % len(other_locations)])
            
        print(f"📍 Distribution: {central_count} Central, {other_count} Mixed")
            
    elif demo_data_enum in [DemoData.SINGAPORE_WIDE, DemoData.LARGE_SCALE]:
        # Island-wide distribution
        central_count = count // 2      # 50% central
        east_count = count // 6         # ~17% east
        west_count = count // 6         # ~17% west  
        north_count = count - central_count - east_count - west_count  # Rest north
        
        print(f"📍 Distribution: {central_count} Central, {east_count} East, {west_count} West, {north_count} North")
        
        # Add locations from each region
        for i in range(central_count):
            locations.append(central_locations[i % len(central_locations)])
        for i in range(east_count):
            locations.append(east_locations[i % len(east_locations)])
        for i in range(west_count):
            locations.append(west_locations[i % len(west_locations)])
        for i in range(north_count):
            locations.append(north_locations[i % len(north_locations)])
    
    else:
        # Fallback - use central locations
        for i in range(count):
            locations.append(central_locations[i % len(central_locations)])
    
    print(f"📍 Generated {len(locations)} total locations")
    
    # Sample coordinates for verification
    if locations:
        print(f"📍 Sample location 1: {locations[0].latitude:.4f}, {locations[0].longitude:.4f}")
        if len(locations) > 1:
            print(f"📍 Sample location 2: {locations[1].latitude:.4f}, {locations[1].longitude:.4f}")
    
    return locations


def generate_demo_data(demo_data_enum: DemoData) -> VehicleRoutePlan:
    """Generate demo data with optimal Singapore coverage."""
    
    print(f"🚛 Generating {demo_data_enum.name} demo data...")
    
    name = f"demo_{demo_data_enum.name.lower()}"
    demo_data = demo_data_enum.value
    random = Random(demo_data.seed)

    demands = ints(random, demo_data.min_demand, demo_data.max_demand + 1)
    service_durations = values(random, SERVICE_DURATION_MINUTES)
    vehicle_capacities = ints(random, demo_data.min_vehicle_capacity, demo_data.max_vehicle_capacity + 1)

    # Generate locations
    all_locations_needed = demo_data.vehicle_count + demo_data.visit_count
    singapore_locations = get_singapore_locations_for_region(demo_data_enum, all_locations_needed)
    
    # Assign to vehicles and visits
    vehicle_locations = singapore_locations[:demo_data.vehicle_count]
    visit_locations = singapore_locations[demo_data.vehicle_count:demo_data.vehicle_count + demo_data.visit_count]

    # Create vehicles
    vehicles = [Vehicle(id=str(i),
                        capacity=next(vehicle_capacities),
                        home_location=vehicle_locations[i],
                        departure_time=datetime.combine(
                            date.today() + timedelta(days=1), demo_data.vehicle_start_time))
                for i in range(demo_data.vehicle_count)]

    # Create visits
    names = generate_names(random)
    visits = []
    
    for i in range(demo_data.visit_count):
        # Simple time windows for testing
        if demo_data_enum in [DemoData.SIMPLE, DemoData.SMALL]:
            # All-day window for simple testing
            min_time = MORNING_WINDOW_START    # 7:00 AM
            max_time = AFTERNOON_WINDOW_END    # 8:00 PM
        else:
            # Flexible time window assignment
            time_choice = random.random()
            if time_choice < 0.6:
                min_time = MORNING_WINDOW_START  # 7:00 AM
                max_time = MORNING_WINDOW_END    # 2:00 PM
            else:
                min_time = AFTERNOON_WINDOW_START  # 12:00 PM
                max_time = AFTERNOON_WINDOW_END    # 8:00 PM
        
        visits.append(Visit(
            id=str(i),
            name=next(names),
            location=visit_locations[i],
            demand=next(demands),
            min_start_time=datetime.combine(date.today() + timedelta(days=1), min_time),
            max_end_time=datetime.combine(date.today() + timedelta(days=1), max_time),
            service_duration=timedelta(minutes=next(service_durations)),
        ))
    
    print(f"📊 Problem: {demo_data.vehicle_count} vehicles, {demo_data.visit_count} visits")
    
    # REMOVED: All distance cache initialization
    # GraphHopper service will handle distance calculations directly
    print(f"✅ Using GraphHopper for real-time distance calculations")

    result = VehicleRoutePlan(name=name,
                            south_west_corner=demo_data.south_west_corner,
                            north_east_corner=demo_data.north_east_corner,
                            vehicles=vehicles,
                            visits=visits)
    
    print(f"✅ Created demo with {len(result.vehicles)} vehicles, {len(result.visits)} visits")
    return result


def tomorrow_at(local_time: time) -> datetime:
    return datetime.combine(date.today(), local_time)