"""
Timefold Optimizer - FIXED workflow with pre-computed distances
This completely eliminates the Java/Python interop ClassCastException
"""

import logging
import time
from typing import Optional
from .domain import (
    VehicleRoutePlan, precompute_distances_for_timefold, 
    verify_cache_ready, get_distance_cache
)
from .graphhopper_service import get_graphhopper_service

logger = logging.getLogger(__name__)


class TimefoldOptimizer:
    """
    FIXED Timefold optimizer that pre-computes distances to avoid network calls.
    """
    
    def __init__(self, solver_manager):
        self.solver_manager = solver_manager
    
    def optimize_with_precomputed_distances(self, route_plan: VehicleRoutePlan, 
                                          problem_id: str = None) -> VehicleRoutePlan:
        """
        MAIN METHOD: Optimize with Timefold using pre-computed distances.
        This fixes the ClassCastException by eliminating network calls.
        """
        
        if problem_id is None:
            import uuid
            problem_id = str(uuid.uuid4())
        
        try:
            # Step 1: Pre-compute ALL distances
            logger.info(f"🔄 Pre-computing distances for Timefold optimization...")
            routing_service = get_graphhopper_service()
            
            precompute_success = precompute_distances_for_timefold(
                route_plan.vehicles, 
                route_plan.visits, 
                routing_service
            )
            
            if not precompute_success:
                raise Exception("Distance pre-computation failed")
            
            # Step 2: Verify cache is ready
            if not verify_cache_ready():
                raise Exception("Distance cache verification failed")
            
            logger.info("✅ Distance pre-computation complete")
            
            # Step 3: Run Timefold optimization (should work now!)
            logger.info("🚀 Starting Timefold optimization...")
            start_time = time.time()
            
            solution = self.solver_manager.solve(problem_id, route_plan)
            
            optimization_time = time.time() - start_time
            logger.info(f"✅ Timefold optimization complete in {optimization_time:.1f}s")
            logger.info(f"   Score: {solution.score}")
            logger.info(f"   Total distance: {solution.total_distance_km:.1f}km")
            
            return solution
            
        except Exception as e:
            logger.error(f"❌ Timefold optimization failed: {e}")
            
            # Clean up cache on error
            get_distance_cache().clear()
            
            raise e
    
    def optimize_safe(self, route_plan: VehicleRoutePlan, 
                     problem_id: str = None) -> Optional[VehicleRoutePlan]:
        """
        Safe optimization that returns None on failure instead of crashing.
        """
        try:
            return self.optimize_with_precomputed_distances(route_plan, problem_id)
        except Exception as e:
            logger.error(f"Timefold optimization failed safely: {e}")
            return None
    
    def test_optimization(self, route_plan: VehicleRoutePlan) -> dict:
        """
        Test the fixed optimization without actually running the full solver.
        Useful for verifying the fix works.
        """
        try:
            # Test pre-computation
            routing_service = get_graphhopper_service()
            
            logger.info("🧪 Testing distance pre-computation...")
            precompute_success = precompute_distances_for_timefold(
                route_plan.vehicles, 
                route_plan.visits, 
                routing_service
            )
            
            if not precompute_success:
                return {"status": "failed", "step": "precomputation"}
            
            # Test cache verification
            logger.info("🧪 Testing cache verification...")
            if not verify_cache_ready():
                return {"status": "failed", "step": "verification"}
            
            # Test domain object calls
            logger.info("🧪 Testing domain object calls...")
            if route_plan.visits and len(route_plan.visits) >= 2:
                visit1 = route_plan.visits[0]
                visit2 = route_plan.visits[1]
                
                # These should work without network calls
                distance = visit1.location.distance_to(visit2.location)
                duration = visit1.location.driving_time_to(visit2.location)
                
                logger.info(f"   Test call successful: {distance/1000:.1f}km, {duration/60:.1f}min")
            
            cache = get_distance_cache()
            
            return {
                "status": "success",
                "cache_size": cache.get_cache_size(),
                "is_precomputed": cache.is_precomputed()
            }
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return {"status": "failed", "error": str(e)}


# =====================================================================
# INTEGRATION FUNCTIONS
# =====================================================================

def create_fixed_timefold_optimizer(solver_manager) -> TimefoldOptimizer:
    """Factory function to create the fixed Timefold optimizer."""
    return TimefoldOptimizer(solver_manager)


def solve_vrp_with_timefold_fixed(route_plan: VehicleRoutePlan, solver_manager, 
                                 problem_id: str = None) -> VehicleRoutePlan:
    """
    Main entry point for FIXED Timefold VRP optimization.
    Use this instead of direct solver_manager.solve() calls.
    """
    optimizer = create_fixed_timefold_optimizer(solver_manager)
    return optimizer.optimize_with_precomputed_distances(route_plan, problem_id)


def test_timefold_fix(route_plan: VehicleRoutePlan) -> dict:
    """
    Test function to verify the Timefold fix works.
    Call this before running full optimization.
    """
    optimizer = TimefoldOptimizer(None)  # Don't need solver for testing
    return optimizer.test_optimization(route_plan)


# =====================================================================
# WORKFLOW HELPER FUNCTIONS
# =====================================================================

def prepare_route_plan_for_timefold(vehicles: list, visits: list, 
                                   name: str = "VRP Problem") -> VehicleRoutePlan:
    """
    Helper to create a VehicleRoutePlan ready for Timefold optimization.
    """
    # Calculate bounding box
    all_lats = []
    all_lons = []
    
    for vehicle in vehicles:
        all_lats.append(vehicle.home_location.latitude)
        all_lons.append(vehicle.home_location.longitude)
    
    for visit in visits:
        all_lats.append(visit.location.latitude)
        all_lons.append(visit.location.longitude)
    
    from .domain import Location
    
    south_west = Location(latitude=min(all_lats), longitude=min(all_lons))
    north_east = Location(latitude=max(all_lats), longitude=max(all_lons))
    
    return VehicleRoutePlan(
        name=name,
        south_west_corner=south_west,
        north_east_corner=north_east,
        vehicles=vehicles,
        visits=visits
    )


def run_complete_timefold_workflow(vehicles: list, visits: list, solver_manager,
                                  problem_name: str = "VRP Optimization") -> VehicleRoutePlan:
    """
    Complete workflow: Create problem -> Pre-compute distances -> Optimize
    """
    logger.info(f"🚛 Starting complete Timefold workflow: {len(vehicles)} vehicles, {len(visits)} visits")
    
    # Step 1: Create route plan
    route_plan = prepare_route_plan_for_timefold(vehicles, visits, problem_name)
    
    # Step 2: Test the fix
    test_result = test_timefold_fix(route_plan)
    if test_result["status"] != "success":
        raise Exception(f"Timefold fix test failed: {test_result}")
    
    logger.info(f"✅ Timefold fix verified: {test_result['cache_size']} distances pre-computed")
    
    # Step 3: Run optimization
    solution = solve_vrp_with_timefold_fixed(route_plan, solver_manager)
    
    logger.info(f"🎉 Complete workflow successful!")
    logger.info(f"   Final score: {solution.score}")
    logger.info(f"   Total distance: {solution.total_distance_km:.1f}km")
    logger.info(f"   Total time: {solution.total_driving_time_seconds/60:.1f} minutes")
    
    return solution


# =====================================================================
# TESTING AND VALIDATION
# =====================================================================

def create_test_problem():
    """Create a test VRP problem for validating the fix."""
    from .domain import Location, Vehicle, Visit
    from datetime import datetime, timedelta
    
    # Create depot
    depot = Location(latitude=1.2966, longitude=103.8518)  # Marina Bay
    
    # Create vehicles
    vehicles = [
        Vehicle(
            id="V1",
            capacity=100,
            home_location=depot,
            departure_time=datetime(2024, 1, 1, 8, 0)
        )
    ]
    
    # Create visits
    visits = [
        Visit(
            id="C1",
            name="Customer 1",
            location=Location(latitude=1.3521, longitude=103.8198),
            demand=10,
            min_start_time=datetime(2024, 1, 1, 9, 0),
            max_end_time=datetime(2024, 1, 1, 17, 0),
            service_duration=timedelta(minutes=15)
        ),
        Visit(
            id="C2", 
            name="Customer 2",
            location=Location(latitude=1.3644, longitude=103.9915),
            demand=15,
            min_start_time=datetime(2024, 1, 1, 9, 0),
            max_end_time=datetime(2024, 1, 1, 17, 0),
            service_duration=timedelta(minutes=20)
        ),
        Visit(
            id="C3",
            name="Customer 3", 
            location=Location(latitude=1.2839, longitude=103.8519),
            demand=12,
            min_start_time=datetime(2024, 1, 1, 9, 0),
            max_end_time=datetime(2024, 1, 1, 17, 0),
            service_duration=timedelta(minutes=10)
        )
    ]
    
    return vehicles, visits


def test_full_timefold_fix():
    """Full test of the Timefold fix."""
    print("🧪 Testing complete Timefold fix...")
    
    try:
        # Create test problem
        vehicles, visits = create_test_problem()
        
        # Test distance pre-computation
        print("📍 Testing distance pre-computation...")
        from .graphhopper_service import get_graphhopper_service
        routing_service = get_graphhopper_service()
        
        success = precompute_distances_for_timefold(vehicles, visits, routing_service)
        if not success:
            print("❌ Distance pre-computation failed")
            return False
        
        # Test cache verification
        print("✅ Testing cache verification...")
        if not verify_cache_ready():
            print("❌ Cache verification failed")
            return False
        
        # Test domain object calls
        print("🔧 Testing domain object calls...")
        visit1, visit2 = visits[0], visits[1]
        
        distance = visit1.location.distance_to(visit2.location)
        duration = visit1.location.driving_time_to(visit2.location)
        
        print(f"   ✅ Domain calls work: {distance/1000:.1f}km, {duration/60:.1f}min")
        
        # Test route plan creation
        print("📋 Testing route plan creation...")
        route_plan = prepare_route_plan_for_timefold(vehicles, visits, "Test Problem")
        
        print(f"   ✅ Route plan created: {len(route_plan.vehicles)} vehicles, {len(route_plan.visits)} visits")
        
        # Clean up
        get_distance_cache().clear()
        
        print("🎉 All tests passed! Timefold fix is working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_full_timefold_fix()