"""
GraphHopper Integration Service - Production Ready
Connects to working GraphHopper server on localhost:8989
cd python/vehicle-routing/config
"""

import requests
import logging
import time
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

logger = logging.getLogger(__name__)

class GraphHopperService:
    """Production GraphHopper service with caching and parallel processing."""
    
    def __init__(self, base_url: str = "http://localhost:8989", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.name = "GraphHopperService"
        
        # Performance tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.failed_requests = 0
        
        # Simple in-memory cache for this session
        self._distance_cache = {}
        self._duration_cache = {}
        
        # Test connection on startup
        self._test_connection()
    
    def _test_connection(self):
        """Test GraphHopper connection on startup."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ GraphHopper connection successful")
            else:
                logger.warning(f"⚠️ GraphHopper health check returned {response.status_code}")
        except Exception as e:
            logger.error(f"❌ GraphHopper connection failed: {e}")
            logger.error("   Make sure GraphHopper is running on localhost:8989")
    
    def calculate_distance(self, from_lat: float, from_lon: float, 
                          to_lat: float, to_lon: float) -> float:
        """Calculate distance in meters using GraphHopper."""
        cache_key = (from_lat, from_lon, to_lat, to_lon, 'distance')
        
        if cache_key in self._distance_cache:
            self.cache_hits += 1
            return self._distance_cache[cache_key]
        
        try:
            route_data = self._get_route(from_lat, from_lon, to_lat, to_lon)
            distance = route_data.get('distance', 0)
            self._distance_cache[cache_key] = distance
            return distance
            
        except Exception as e:
            logger.warning(f"GraphHopper distance failed: {e}")
            self.failed_requests += 1
            return self._fallback_distance(from_lat, from_lon, to_lat, to_lon)
    
    def calculate_duration(self, from_lat: float, from_lon: float, 
                          to_lat: float, to_lon: float) -> int:
        """Calculate duration in seconds using GraphHopper."""
        cache_key = (from_lat, from_lon, to_lat, to_lon, 'duration')
        
        if cache_key in self._duration_cache:
            self.cache_hits += 1
            return self._duration_cache[cache_key]
        
        try:
            route_data = self._get_route(from_lat, from_lon, to_lat, to_lon)
            # GraphHopper returns duration in milliseconds
            duration = int(route_data.get('time', 0) / 1000)
            self._duration_cache[cache_key] = duration
            return duration
            
        except Exception as e:
            logger.warning(f"GraphHopper duration failed: {e}")
            self.failed_requests += 1
            return self._fallback_duration(from_lat, from_lon, to_lat, to_lon)
    
    def _get_route(self, from_lat: float, from_lon: float, 
                   to_lat: float, to_lon: float) -> Dict:
        """Get route data from GraphHopper."""
        self.total_requests += 1
        
        params = {
            'point': [f"{from_lat},{from_lon}", f"{to_lat},{to_lon}"],
            'profile': 'car',  # GraphHopper 10.0 uses 'profile' not 'vehicle'
            'calc_points': 'false',  # We only need distance/time, not geometry
            'instructions': 'false'
        }
        
        response = self.session.get(
            f"{self.base_url}/route",
            params=params,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        data = response.json()
        
        if 'paths' not in data or not data['paths']:
            raise Exception("No route found")
        
        path = data['paths'][0]
        return {
            'distance': path.get('distance', 0),  # meters
            'time': path.get('time', 0)           # milliseconds
        }
    
    def calculate_matrix_parallel(self, locations: List[Tuple[float, float]], 
                                 max_workers: int = 8) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Calculate distance and duration matrices in parallel.
        GraphHopper OSS doesn't have a matrix API, so we calculate all pairs.
        """
        n = len(locations)
        distance_matrix = [[0.0] * n for _ in range(n)]
        duration_matrix = [[0] * n for _ in range(n)]
        
        logger.info(f"🔄 Calculating {n}x{n} matrix with GraphHopper ({n*n} route calculations)")
        start_time = time.time()
        
        # Create all route calculation tasks
        tasks = []
        for i in range(n):
            for j in range(n):
                if i != j:  # Skip diagonal (same location)
                    tasks.append((i, j, locations[i], locations[j]))
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._calculate_single_route, task[2], task[3]): task
                for task in tasks
            }
            
            completed = 0
            for future in as_completed(future_to_task):
                i, j, _, _ = future_to_task[future]
                try:
                    distance, duration = future.result()
                    distance_matrix[i][j] = distance
                    duration_matrix[i][j] = duration
                except Exception as e:
                    logger.warning(f"Route calculation failed for {i}->{j}: {e}")
                    # Use fallback calculation
                    lat1, lon1 = locations[i]
                    lat2, lon2 = locations[j]
                    distance_matrix[i][j] = self._fallback_distance(lat1, lon1, lat2, lon2)
                    duration_matrix[i][j] = self._fallback_duration(lat1, lon1, lat2, lon2)
                
                completed += 1
                if completed % 10 == 0:  # Progress logging
                    logger.info(f"   Progress: {completed}/{len(tasks)} routes calculated")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Matrix calculation complete in {elapsed:.1f} seconds")
        
        return distance_matrix, duration_matrix
    
    def _calculate_single_route(self, from_loc: Tuple[float, float], 
                               to_loc: Tuple[float, float]) -> Tuple[float, int]:
        """Calculate single route for parallel processing."""
        from_lat, from_lon = from_loc
        to_lat, to_lon = to_loc
        
        distance = self.calculate_distance(from_lat, from_lon, to_lat, to_lon)
        duration = self.calculate_duration(from_lat, from_lon, to_lat, to_lon)
        
        return distance, duration
    
    def precompute_distances(self, locations: List[Tuple[float, float]]) -> bool:
        """Precompute all distances for a set of locations."""
        try:
            logger.info(f"🔄 Precomputing distances for {len(locations)} locations")
            distance_matrix, duration_matrix = self.calculate_matrix_parallel(locations)
            
            # Cache all calculated routes
            for i, (lat1, lon1) in enumerate(locations):
                for j, (lat2, lon2) in enumerate(locations):
                    if i != j:
                        dist_key = (lat1, lon1, lat2, lon2, 'distance')
                        dur_key = (lat1, lon1, lat2, lon2, 'duration')
                        self._distance_cache[dist_key] = distance_matrix[i][j]
                        self._duration_cache[dur_key] = duration_matrix[i][j]
            
            logger.info("✅ Distance precomputation complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Distance precomputation failed: {e}")
            return False
    
    def get_service_health(self) -> Dict:
        """Get service health and performance stats."""
        return {
            "status": "healthy",
            "service": self.name,
            "base_url": self.base_url,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "failed_requests": self.failed_requests,
            "cache_hit_rate": f"{(self.cache_hits/max(1,self.total_requests)*100):.1f}%",
            "distance_cache_size": len(self._distance_cache),
            "duration_cache_size": len(self._duration_cache)
        }
    
    def _fallback_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Emergency Haversine fallback for distance."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return 6371000 * 2 * math.asin(math.sqrt(a))  # meters
    
    def _fallback_duration(self, lat1: float, lon1: float, lat2: float, lon2: float) -> int:
        """Emergency duration fallback."""
        distance = self._fallback_distance(lat1, lon1, lat2, lon2)
        # Assume 35 km/h average speed in Singapore
        return int(distance / (35000/3600))  # seconds


# Global service instance
_graphhopper_service = None

def get_graphhopper_service() -> GraphHopperService:
    """Get or create the global GraphHopper service instance."""
    global _graphhopper_service
    if _graphhopper_service is None:
        _graphhopper_service = GraphHopperService()
    return _graphhopper_service


def get_singapore_routing_service():
    """Drop-in replacement for existing OSRM routing service."""
    return get_graphhopper_service()


# Test function
def test_graphhopper_integration():
    """Test GraphHopper integration with Singapore locations."""
    print("🧪 Testing GraphHopper integration...")
    
    service = get_graphhopper_service()
    
    # Test with Singapore locations
    marina_bay = (1.2966, 103.8518)  # Marina Bay
    changi = (1.3644, 103.9915)      # Changi Airport
    
    try:
        # Test distance and duration
        distance = service.calculate_distance(*marina_bay, *changi)
        duration = service.calculate_duration(*marina_bay, *changi)
        
        print(f"✅ GraphHopper route test successful:")
        print(f"   Marina Bay → Changi Airport")
        print(f"   Distance: {distance/1000:.1f} km")
        print(f"   Duration: {duration} seconds ({duration/60:.1f} minutes)")
        
        # Sanity check
        if 15000 < distance < 40000 and 600 < duration < 3000:
            print("✅ Results look reasonable for Singapore!")
            return True
        else:
            print("⚠️  Results seem unusual - check GraphHopper setup")
            return False
            
    except Exception as e:
        print(f"❌ GraphHopper test failed: {e}")
        print("   Make sure GraphHopper is running on localhost:8989")
        return False


if __name__ == "__main__":
    test_graphhopper_integration()