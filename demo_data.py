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
    """
    FIXED: Demo data categories that match frontend expectations exactly.
    
    Frontend expects these specific enum names:
    - SIMPLE, SMALL (for testing)
    - SINGAPORE_CENTRAL, SINGAPORE_EAST, SINGAPORE_WEST, SINGAPORE_NORTH (regional)
    - SINGAPORE_WIDE (island-wide)
    - CENTRAL (backward compatibility)
    - LARGE_SCALE (stress testing)
    """
    
    # ===== TESTING CATEGORIES =====
    
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

    # ===== PRODUCTION CATEGORIES (FRONTEND EXPECTS THESE EXACT NAMES) =====
    
    # Singapore Central - Marina Bay, CBD, Orchard area
    SINGAPORE_CENTRAL = _DemoDataProperties(2001, 15, 2, time(8, 0),
                                          1, 2, 15, 30,
                                          Location(latitude=1.2700, longitude=103.8200),
                                          Location(latitude=1.3000, longitude=103.8700))
    
    # Singapore East - Changi, Tampines, Bedok area
    SINGAPORE_EAST = _DemoDataProperties(2002, 15, 2, time(8, 0),
                                       1, 2, 15, 30,
                                       Location(latitude=1.3200, longitude=103.9200),
                                       Location(latitude=1.3800, longitude=104.0000))
    
    # Singapore West - Jurong, Boon Lay, Pioneer area
    SINGAPORE_WEST = _DemoDataProperties(2003, 15, 2, time(8, 0),
                                       1, 2, 15, 30,
                                       Location(latitude=1.3000, longitude=103.6500),
                                       Location(latitude=1.3600, longitude=103.7800))
    
    # Singapore North - Woodlands, Yishun, Ang Mo Kio area
    SINGAPORE_NORTH = _DemoDataProperties(2004, 15, 2, time(8, 0),
                                        1, 2, 15, 30,
                                        Location(latitude=1.4000, longitude=103.7500),
                                        Location(latitude=1.4500, longitude=103.8800))

    # ===== BACKWARD COMPATIBILITY =====
    
    # Keep CENTRAL for backward compatibility (maps to SINGAPORE_CENTRAL)
    CENTRAL = _DemoDataProperties(2001, 15, 2, time(8, 0),
                                  1, 2, 15, 30,
                                  Location(latitude=1.2700, longitude=103.8200),
                                  Location(latitude=1.3000, longitude=103.8700))

    # ===== ISLAND-WIDE AND STRESS TESTING =====
    
    # Singapore Island-Wide - CRITICAL for iterative testing
    SINGAPORE_WIDE = _DemoDataProperties(3001, 30, 3, time(8, 0),
                                         1, 4, 20, 40,
                                         Location(latitude=1.2400, longitude=103.6000),
                                         Location(latitude=1.4700, longitude=104.0200))

    # Large Scale Test - For future stress testing
    LARGE_SCALE = _DemoDataProperties(4001, 50, 5, time(8, 0),
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
    """
    FIXED: Optimized location distribution with guaranteed coverage matching frontend expectations.
    
    This function now properly maps to the regions expected by the frontend:
    - SINGAPORE_CENTRAL: Marina Bay, CBD, Orchard
    - SINGAPORE_EAST: Changi, Tampines, Bedok  
    - SINGAPORE_WEST: Jurong, Boon Lay, Pioneer
    - SINGAPORE_NORTH: Woodlands, Yishun, Ang Mo Kio
    - SINGAPORE_WIDE: Island-wide distribution
    """
    
    print(f"📍 Generating {count} locations for {demo_data_enum.name}")
    
    # FIXED: Enhanced strategic locations for maximum testing coverage
    
    # Central Singapore - Primary business and tourist areas
    central_locations = [
        Location(latitude=1.2966, longitude=103.8518),  # Marina Bay Sands
        Location(latitude=1.2839, longitude=103.8519),  # Raffles Place
        Location(latitude=1.3048, longitude=103.8318),  # Orchard Road
        Location(latitude=1.2792, longitude=103.8480),  # Chinatown
        Location(latitude=1.3521, longitude=103.8198),  # City Hall
        Location(latitude=1.3000, longitude=103.8500),  # Bugis
        Location(latitude=1.2800, longitude=103.8400),  # Clarke Quay
        Location(latitude=1.2900, longitude=103.8600),  # Marina Centre
        Location(latitude=1.2966, longitude=103.8518),  # Marina Bay (repeat for density)
        Location(latitude=1.2918, longitude=103.8456),  # Boat Quay
    ]
    
    # East Singapore - Residential and commercial hubs
    east_locations = [
        Location(latitude=1.3644, longitude=103.9915),  # Changi Airport
        Location(latitude=1.3496, longitude=103.9568),  # Tampines Mall
        Location(latitude=1.3329, longitude=103.9436),  # Bedok Mall
        Location(latitude=1.3240, longitude=103.9520),  # Tanah Merah MRT
        Location(latitude=1.3558, longitude=103.9449),  # Pasir Ris
        Location(latitude=1.3376, longitude=103.9625),  # Simei
        Location(latitude=1.3194, longitude=103.9443),  # Kembangan
        Location(latitude=1.3271, longitude=103.9240),  # Eunos
    ]
    
    # West Singapore - Industrial and residential areas
    west_locations = [
        Location(latitude=1.3387, longitude=103.7053),  # Jurong East
        Location(latitude=1.3553, longitude=103.6874),  # Boon Lay
        Location(latitude=1.3477, longitude=103.7430),  # Jurong West
        Location(latitude=1.3200, longitude=103.6800),  # Pioneer
        Location(latitude=1.3352, longitude=103.7421),  # Clementi
        Location(latitude=1.3464, longitude=103.6800),  # Tuas
        Location(latitude=1.3513, longitude=103.7065),  # Lakeside
        Location(latitude=1.3389, longitude=103.7454),  # Chinese Garden
    ]
    
    # North Singapore - Residential and nature areas
    north_locations = [
        Location(latitude=1.4382, longitude=103.7890),  # Woodlands
        Location(latitude=1.4294, longitude=103.8356),  # Yishun
        Location(latitude=1.4168, longitude=103.8432),  # Ang Mo Kio
        Location(latitude=1.3966, longitude=103.8467),  # Bishan
        Location(latitude=1.4241, longitude=103.8315),  # Khatib
        Location(latitude=1.4491, longitude=103.8222),  # Sembawang
        Location(latitude=1.4304, longitude=103.7703),  # Admiralty
        Location(latitude=1.4042, longitude=103.8354),  # Thomson
    ]
    
    locations = []
    
    # FIXED: Distribution logic matching frontend expectations
    
    if demo_data_enum in [DemoData.SIMPLE, DemoData.SMALL]:
        # Simple testing - use only central locations for predictable results
        for i in range(count):
            locations.append(central_locations[i % len(central_locations)])
        print(f"📍 Distribution: {count} Central (simple test)")
    
    elif demo_data_enum in [DemoData.CENTRAL, DemoData.SINGAPORE_CENTRAL]:
        # Central Singapore - mostly CBD and Marina Bay area
        central_count = max(1, count * 4 // 5)  # 80% central
        other_count = count - central_count
        
        for i in range(central_count):
            locations.append(central_locations[i % len(central_locations)])
        
        # Add some variety from nearby areas
        nearby_locations = east_locations[:3] + west_locations[:2]  # Limited to nearby areas
        for i in range(other_count):
            locations.append(nearby_locations[i % len(nearby_locations)])
            
        print(f"📍 Distribution: {central_count} Central, {other_count} Nearby")
    
    elif demo_data_enum == DemoData.SINGAPORE_EAST:
        # East Singapore - focus on eastern areas
        east_count = max(1, count * 3 // 4)  # 75% east
        central_count = count - east_count   # 25% central (for connectivity)
        
        for i in range(east_count):
            locations.append(east_locations[i % len(east_locations)])
        for i in range(central_count):
            locations.append(central_locations[i % len(central_locations)])
            
        print(f"📍 Distribution: {east_count} East, {central_count} Central")
    
    elif demo_data_enum == DemoData.SINGAPORE_WEST:
        # West Singapore - focus on western areas
        west_count = max(1, count * 3 // 4)  # 75% west
        central_count = count - west_count   # 25% central (for connectivity)
        
        for i in range(west_count):
            locations.append(west_locations[i % len(west_locations)])
        for i in range(central_count):
            locations.append(central_locations[i % len(central_locations)])
            
        print(f"📍 Distribution: {west_count} West, {central_count} Central")
    
    elif demo_data_enum == DemoData.SINGAPORE_NORTH:
        # North Singapore - focus on northern areas
        north_count = max(1, count * 3 // 4)  # 75% north
        central_count = count - north_count   # 25% central (for connectivity)
        
        for i in range(north_count):
            locations.append(north_locations[i % len(north_locations)])
        for i in range(central_count):
            locations.append(central_locations[i % len(central_locations)])
            
        print(f"📍 Distribution: {north_count} North, {central_count} Central")
            
    elif demo_data_enum in [DemoData.SINGAPORE_WIDE, DemoData.LARGE_SCALE]:
        # Island-wide distribution - CRITICAL for iterative testing variance
        central_count = count // 3      # ~33% central (main hub)
        east_count = count // 4         # ~25% east
        west_count = count // 4         # ~25% west  
        north_count = count - central_count - east_count - west_count  # Rest north
        
        print(f"📍 Distribution: {central_count} Central, {east_count} East, {west_count} West, {north_count} North")
        
        # Ensure good island-wide spread for variance testing
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
        print(f"📍 Distribution: {count} Central (fallback)")
    
    print(f"📍 Generated {len(locations)} total locations")
    
    # Sample coordinates for verification
    if locations:
        print(f"📍 Sample location 1: {locations[0].latitude:.4f}, {locations[0].longitude:.4f}")
        if len(locations) > 1:
            print(f"📍 Sample location 2: {locations[1].latitude:.4f}, {locations[1].longitude:.4f}")
        if len(locations) > 5:
            print(f"📍 Sample location 6: {locations[5].latitude:.4f}, {locations[5].longitude:.4f}")
    
    return locations


def generate_demo_data(demo_data_enum: DemoData) -> VehicleRoutePlan:
    """
    FIXED: Generate demo data with optimal Singapore coverage matching frontend categories.
    
    This function now ensures perfect alignment between frontend and backend demo categories.
    """
    
    print(f"🚛 Generating {demo_data_enum.name} demo data...")
    
    name = f"demo_{demo_data_enum.name.lower()}"
    demo_data = demo_data_enum.value
    random = Random(demo_data.seed)

    demands = ints(random, demo_data.min_demand, demo_data.max_demand + 1)
    service_durations = values(random, SERVICE_DURATION_MINUTES)
    vehicle_capacities = ints(random, demo_data.min_vehicle_capacity, demo_data.max_vehicle_capacity + 1)

    # Generate locations with region-specific distribution
    all_locations_needed = demo_data.vehicle_count + demo_data.visit_count
    singapore_locations = get_singapore_locations_for_region(demo_data_enum, all_locations_needed)
    
    # Assign to vehicles and visits
    vehicle_locations = singapore_locations[:demo_data.vehicle_count]
    visit_locations = singapore_locations[demo_data.vehicle_count:demo_data.vehicle_count + demo_data.visit_count]

    # Create vehicles with proper home locations
    vehicles = [Vehicle(id=str(i),
                        capacity=next(vehicle_capacities),
                        home_location=vehicle_locations[i],
                        departure_time=datetime.combine(
                            date.today() + timedelta(days=1), demo_data.vehicle_start_time))
                for i in range(demo_data.vehicle_count)]

    # Create visits with appropriate time windows
    names = generate_names(random)
    visits = []
    
    for i in range(demo_data.visit_count):
        # FIXED: Time window assignment for better testing
        if demo_data_enum in [DemoData.SIMPLE, DemoData.SMALL]:
            # All-day window for simple testing
            min_time = MORNING_WINDOW_START    # 7:00 AM
            max_time = AFTERNOON_WINDOW_END    # 8:00 PM
        else:
            # Variable time window assignment for realistic scenarios
            time_choice = random.random()
            if time_choice < 0.5:
                # Morning delivery window
                min_time = MORNING_WINDOW_START  # 7:00 AM
                max_time = MORNING_WINDOW_END    # 2:00 PM
            elif time_choice < 0.8:
                # Afternoon delivery window
                min_time = AFTERNOON_WINDOW_START  # 12:00 PM
                max_time = AFTERNOON_WINDOW_END    # 8:00 PM
            else:
                # Extended day window
                min_time = MORNING_WINDOW_START    # 7:00 AM
                max_time = EVENING_WINDOW_END      # 10:00 PM
        
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
    print(f"📍 Region: {demo_data_enum.name}")
    
    # Using GraphHopper for real-time distance calculations
    print(f"✅ Using GraphHopper for real-time distance calculations")

    result = VehicleRoutePlan(name=name,
                            south_west_corner=demo_data.south_west_corner,
                            north_east_corner=demo_data.north_east_corner,
                            vehicles=vehicles,
                            visits=visits)
    
    print(f"✅ Created {demo_data_enum.name} demo with {len(result.vehicles)} vehicles, {len(result.visits)} visits")
    
    # Verification for critical categories
    if demo_data_enum == DemoData.SINGAPORE_WIDE:
        print("🔍 SINGAPORE_WIDE verification: Ready for iterative consistency testing")
    elif demo_data_enum in [DemoData.SINGAPORE_CENTRAL, DemoData.SINGAPORE_EAST, 
                           DemoData.SINGAPORE_WEST, DemoData.SINGAPORE_NORTH]:
        print(f"🔍 {demo_data_enum.name} verification: Regional demo ready for frontend")
    
    return result


def tomorrow_at(local_time: time) -> datetime:
    return datetime.combine(date.today(), local_time)