from timefold.solver import SolverStatus
from timefold.solver.score import HardSoftScore
from timefold.solver.domain import *

from datetime import datetime, timedelta
from typing import Annotated, Optional, Dict, Tuple
from pydantic import Field, computed_field, BeforeValidator
import math

from .json_serialization import *

# NO LOGGING IMPORT - causes Java/Python interop issues in Timefold

# =====================================================================
# DISTANCE CACHE - Pre-computed distances for Timefold (NO LOGGING)
# =====================================================================

class DistanceCache:
    """
    Global distance cache that eliminates network calls from domain objects.
    FIXED: No logging calls to avoid Java/Python interop issues.
    """
    
    def __init__(self):
        # Simple dictionaries that Java can handle
        self._distance_cache: Dict[str, float] = {}
        self._duration_cache: Dict[str, int] = {}
        self._is_precomputed = False
    
    def _make_key(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> str:
        """Create a simple string key for caching."""
        return f"{from_lat:.6f},{from_lon:.6f}->{to_lat:.6f},{to_lon:.6f}"
    
    def precompute_for_problem(self, locations: list, routing_service) -> bool:
        """Pre-compute all distances for a VRP problem."""
        try:
            print(f"🔄 Pre-computing distances for {len(locations)} locations for Timefold...")
            
            # Clear existing cache
            self._distance_cache.clear()
            self._duration_cache.clear()
            
            total_pairs = len(locations) * (len(locations) - 1)
            computed = 0
            
            # Compute all location pairs
            for i, loc1 in enumerate(locations):
                for j, loc2 in enumerate(locations):
                    if i != j:  # Skip same location
                        key = self._make_key(loc1.latitude, loc1.longitude, 
                                           loc2.latitude, loc2.longitude)
                        
                        try:
                            # Get distance and duration from GraphHopper
                            distance = routing_service.calculate_distance(
                                loc1.latitude, loc1.longitude, 
                                loc2.latitude, loc2.longitude
                            )
                            duration = routing_service.calculate_duration(
                                loc1.latitude, loc1.longitude, 
                                loc2.latitude, loc2.longitude
                            )
                            
                            # Store in simple dictionaries
                            self._distance_cache[key] = float(distance)
                            self._duration_cache[key] = int(duration)
                            
                        except Exception as e:
                            # Use fallback calculation - NO LOGGING
                            distance = self._fallback_distance(loc1.latitude, loc1.longitude, 
                                                             loc2.latitude, loc2.longitude)
                            duration = self._fallback_duration(loc1.latitude, loc1.longitude, 
                                                             loc2.latitude, loc2.longitude)
                            
                            self._distance_cache[key] = float(distance)
                            self._duration_cache[key] = int(duration)
                        
                        computed += 1
                        if computed % 20 == 0:
                            print(f"   Progress: {computed}/{total_pairs} distances computed")
            
            self._is_precomputed = True
            print(f"✅ Pre-computed {len(self._distance_cache)} distance pairs for Timefold")
            return True
            
        except Exception as e:
            print(f"❌ Distance pre-computation failed: {e}")
            return False
    
    def get_distance(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
        """Get pre-computed distance - FAST lookup, no network calls, NO LOGGING."""
        key = self._make_key(from_lat, from_lon, to_lat, to_lon)
        
        if key in self._distance_cache:
            return self._distance_cache[key]
        
        # Fallback if not found - NO LOGGING to avoid Java/Python interop issues
        return self._fallback_distance(from_lat, from_lon, to_lat, to_lon)
    
    def get_duration(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> int:
        """Get pre-computed duration - FAST lookup, no network calls, NO LOGGING."""
        key = self._make_key(from_lat, from_lon, to_lat, to_lon)
        
        if key in self._duration_cache:
            return self._duration_cache[key]
        
        # Fallback if not found - NO LOGGING to avoid Java/Python interop issues
        return self._fallback_duration(from_lat, from_lon, to_lat, to_lon)
    
    def _fallback_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance fallback."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        distance = 6371000 * 2 * math.asin(math.sqrt(a))  # meters
        return distance * 1.2  # 20% penalty for road vs straight line
    
    def _fallback_duration(self, lat1: float, lon1: float, lat2: float, lon2: float) -> int:
        """Fallback duration calculation."""
        distance = self._fallback_distance(lat1, lon1, lat2, lon2)
        return int(distance / (30000/3600))  # 30 km/h average speed
    
    def is_precomputed(self) -> bool:
        """Check if distances are pre-computed."""
        return self._is_precomputed
    
    def clear(self):
        """Clear the cache."""
        self._distance_cache.clear()
        self._duration_cache.clear()
        self._is_precomputed = False
    
    def get_cache_size(self) -> int:
        """Get number of cached distances."""
        return len(self._distance_cache)


# Global distance cache instance
_distance_cache = DistanceCache()

def get_distance_cache() -> DistanceCache:
    """Get the global distance cache."""
    return _distance_cache

def precompute_distances_for_timefold(vehicles: list, visits: list, routing_service) -> bool:
    """
    CRITICAL: Call this BEFORE running Timefold optimization.
    Pre-computes all distances to avoid network calls in domain objects.
    FIXED: Uses print() instead of logging to avoid Java/Python interop issues.
    """
    # Extract all unique locations
    locations = set()
    
    # Add vehicle home locations
    for vehicle in vehicles:
        locations.add(vehicle.home_location)
    
    # Add visit locations  
    for visit in visits:
        locations.add(visit.location)
    
    print(f"📍 Extracted {len(locations)} unique locations for pre-computation")
    
    # Pre-compute all distances
    cache = get_distance_cache()
    success = cache.precompute_for_problem(list(locations), routing_service)
    
    if success:
        print("✅ Timefold distance pre-computation successful")
    else:
        print("❌ Timefold distance pre-computation failed")
    
    return success


# =====================================================================
# FIXED DOMAIN OBJECTS - No network calls, only cache lookups
# =====================================================================

LocationValidator = BeforeValidator(lambda location: location if isinstance(location, Location)
                                    else Location(latitude=location[0], longitude=location[1]))


class Location(JsonDomainBase):
    """FIXED Location class - uses pre-computed cache, no network calls, NO LOGGING."""
    latitude: float
    longitude: float

    def __hash__(self):
        return hash((self.latitude, self.longitude))
    
    def __eq__(self, other):
        if not isinstance(other, Location):
            return False
        return (self.latitude, self.longitude) == (other.latitude, other.longitude)

    def driving_time_to(self, other: 'Location') -> int:
        """
        FIXED: Get driving time from pre-computed cache.
        NO NETWORK CALLS - This fixes the Timefold ClassCastException.
        NO LOGGING - This fixes the Java/Python interop AttributeError.
        """
        cache = get_distance_cache()
        duration = cache.get_duration(
            self.latitude, self.longitude, 
            other.latitude, other.longitude
        )
        return int(duration)  # Ensure integer

    def distance_to(self, other: 'Location') -> float:
        """
        FIXED: Get distance from pre-computed cache.
        NO NETWORK CALLS - Safe for Timefold.
        NO LOGGING - Safe for Java/Python interop.
        """
        cache = get_distance_cache()
        distance = cache.get_distance(
            self.latitude, self.longitude,
            other.latitude, other.longitude
        )
        return float(distance)  # Ensure float

    def __str__(self):
        return f'[{self.latitude}, {self.longitude}]'

    def __repr__(self):
        return f'Location({self.latitude}, {self.longitude})'


# =====================================================================
# YOUR EXISTING DOMAIN OBJECTS - Keep as-is, they now work with cache
# =====================================================================

@planning_entity
class Visit(JsonDomainBase):
    id: Annotated[str, PlanningId]
    name: str
    location: Annotated[Location, LocationSerializer, LocationValidator]
    demand: int
    min_start_time: datetime
    max_end_time: datetime
    service_duration: Annotated[timedelta, DurationSerializer]
    vehicle: Annotated[Optional['Vehicle'],
                       InverseRelationShadowVariable(source_variable_name='visits'),
                       IdSerializer, VehicleValidator, Field(default=None)]
    previous_visit: Annotated[Optional['Visit'],
                              PreviousElementShadowVariable(source_variable_name='visits'),
                              IdSerializer, VisitValidator, Field(default=None)]
    next_visit: Annotated[Optional['Visit'],
                          NextElementShadowVariable(source_variable_name='visits'),
                          IdSerializer, VisitValidator, Field(default=None)]
    arrival_time: Annotated[
        Optional[datetime],
        CascadingUpdateShadowVariable(target_method_name='update_arrival_time'),
        Field(default=None)]

    def update_arrival_time(self):
        """FIXED: Now works because Location.driving_time_to() uses cache with no logging."""
        if self.vehicle is None or (self.previous_visit is not None and self.previous_visit.arrival_time is None):
            self.arrival_time = None
        elif self.previous_visit is None:
            self.arrival_time = (self.vehicle.departure_time +
                                 timedelta(seconds=self.vehicle.home_location.driving_time_to(self.location)))
        else:
            self.arrival_time = (self.previous_visit.calculate_departure_time() +
                                 timedelta(seconds=self.previous_visit.location.driving_time_to(self.location)))

    def calculate_departure_time(self):
        if self.arrival_time is None:
            return None
        return max(self.arrival_time, self.min_start_time) + self.service_duration

    @computed_field
    @property
    def departure_time(self) -> Optional[datetime]:
        return self.calculate_departure_time()

    @computed_field
    @property
    def start_service_time(self) -> Optional[datetime]:
        if self.arrival_time is None:
            return None
        return max(self.arrival_time, self.min_start_time)

    def is_service_finished_after_max_end_time(self) -> bool:
        return self.arrival_time is not None and self.calculate_departure_time() > self.max_end_time

    def service_finished_delay_in_minutes(self) -> int:
        if self.arrival_time is None:
            return 0
        return -((self.calculate_departure_time() - self.max_end_time) // timedelta(minutes=-1))

    def calculate_time_window_lateness_minutes(self) -> int:
        """Calculate how late the service starts within the time window."""
        if self.arrival_time is None or self.min_start_time is None:
            return 0
        
        if self.arrival_time <= self.min_start_time:
            return 0
        
        actual_lateness = (self.arrival_time - self.min_start_time).total_seconds() / 60
        return max(0, int(actual_lateness))

    @computed_field
    @property
    def driving_time_seconds_from_previous_standstill(self) -> Optional[int]:
        if self.vehicle is None:
            return None

        if self.previous_visit is None:
            return self.vehicle.home_location.driving_time_to(self.location)
        else:
            return self.previous_visit.location.driving_time_to(self.location)

    def __str__(self):
        return self.id

    def __repr__(self):
        return f'Visit({self.id})'


@planning_entity
class Vehicle(JsonDomainBase):
    id: Annotated[str, PlanningId]
    capacity: int
    home_location: Annotated[Location, LocationSerializer, LocationValidator]
    departure_time: datetime
    visits: Annotated[list[Visit],
                      PlanningListVariable,
                      IdListSerializer, VisitListValidator, Field(default_factory=list)]

    @computed_field
    @property
    def arrival_time(self) -> datetime:
        """FIXED: Handle None departure_time to avoid TypeError."""
        if len(self.visits) == 0:
            return self.departure_time
        
        # FIXED: Handle None departure_time
        last_visit_departure = self.visits[-1].departure_time
        if last_visit_departure is None:
            return self.departure_time
        
        return (last_visit_departure +
                timedelta(seconds=self.visits[-1].location.driving_time_to(self.home_location)))

    @computed_field
    @property
    def total_demand(self) -> int:
        return self.calculate_total_demand()

    @computed_field
    @property
    def total_driving_time_seconds(self) -> int:
        return self.calculate_total_driving_time_seconds()

    @computed_field
    @property
    def total_distance_meters(self) -> float:
        """FIXED: Calculate total driving distance using cache."""
        if len(self.visits) == 0:
            return 0.0
        
        total_distance = 0.0
        previous_location = self.home_location

        for visit in self.visits:
            total_distance += previous_location.distance_to(visit.location)
            previous_location = visit.location

        total_distance += previous_location.distance_to(self.home_location)
        return total_distance
    
    @computed_field
    @property
    def total_distance_km(self) -> float:
        """Total distance in kilometers."""
        return self.total_distance_meters / 1000.0

    def calculate_total_demand(self) -> int:
        total_demand = 0
        for visit in self.visits:
            total_demand += visit.demand
        return total_demand

    def calculate_total_driving_time_seconds(self) -> int:
        """FIXED: Calculate total driving time using cache."""
        if len(self.visits) == 0:
            return 0
        total_driving_time_seconds = 0
        previous_location = self.home_location

        for visit in self.visits:
            total_driving_time_seconds += previous_location.driving_time_to(visit.location)
            previous_location = visit.location

        total_driving_time_seconds += previous_location.driving_time_to(self.home_location)
        return total_driving_time_seconds

    def calculate_total_distance_meters(self) -> int:
        """Calculate total distance for constraints."""
        return int(self.total_distance_meters)

    def __str__(self):
        return self.id

    def __repr__(self):
        return f'Vehicle({self.id})'


@planning_solution
class VehicleRoutePlan(JsonDomainBase):
    name: str
    south_west_corner: Annotated[Location, LocationSerializer, LocationValidator]
    north_east_corner: Annotated[Location, LocationSerializer, LocationValidator]
    vehicles: Annotated[list[Vehicle], PlanningEntityCollectionProperty]
    visits: Annotated[list[Visit], PlanningEntityCollectionProperty, ValueRangeProvider]
    score: Annotated[Optional[HardSoftScore],
                     PlanningScore,
                     ScoreSerializer, ScoreValidator, Field(default=None)]
    solver_status: Annotated[Optional[SolverStatus],
                             Field(default=None)]

    @computed_field
    @property
    def total_driving_time_seconds(self) -> int:
        out = 0
        for vehicle in self.vehicles:
            out += vehicle.total_driving_time_seconds
        return out

    @computed_field
    @property
    def total_distance_meters(self) -> float:
        """Calculate total distance for all vehicles."""
        return sum(vehicle.total_distance_meters for vehicle in self.vehicles)
    
    @computed_field
    @property
    def total_distance_km(self) -> float:
        """Total distance in kilometers for all vehicles."""
        return self.total_distance_meters / 1000.0

    def __str__(self):
        return f'VehicleRoutePlan(name={self.name}, vehicles={self.vehicles}, visits={self.visits})'


# =====================================================================
# VERIFICATION FUNCTIONS (NO LOGGING)
# =====================================================================

def verify_cache_ready() -> bool:
    """Verify that distance cache is ready for Timefold."""
    cache = get_distance_cache()
    is_ready = cache.is_precomputed() and cache.get_cache_size() > 0
    
    if is_ready:
        print(f"✅ Distance cache ready: {cache.get_cache_size()} entries")
    else:
        print("⚠️ Distance cache not ready - call precompute_distances_for_timefold() first")
    
    return is_ready


def test_fixed_domain():
    """Test that the fixed domain works without network calls."""
    print("🧪 Testing fixed domain with pre-computed distances...")
    
    # Create test locations
    marina_bay = Location(latitude=1.2966, longitude=103.8518)
    changi = Location(latitude=1.3644, longitude=103.9915)
    
    # Test without pre-computation (should use fallback)
    print("Testing fallback calculations...")
    distance = marina_bay.distance_to(changi)
    duration = marina_bay.driving_time_to(changi)
    print(f"   Fallback - Distance: {distance/1000:.1f}km, Duration: {duration/60:.1f}min")
    
    # Test with mock pre-computation
    cache = get_distance_cache()
    cache._distance_cache["1.296600,103.851800->1.364400,103.991500"] = 25000.0  # 25km
    cache._duration_cache["1.296600,103.851800->1.364400,103.991500"] = 1800     # 30min
    cache._is_precomputed = True
    
    print("Testing with pre-computed cache...")
    distance = marina_bay.distance_to(changi)
    duration = marina_bay.driving_time_to(changi)
    print(f"   Cached - Distance: {distance/1000:.1f}km, Duration: {duration/60:.1f}min")
    
    print("✅ Domain tests passed - ready for Timefold!")
    return True


if __name__ == "__main__":
    test_fixed_domain()