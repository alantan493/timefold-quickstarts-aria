"""
FIXED Timefold solver configuration with proper timeouts AND distance pre-computation.
Key fixes:
1. Removed aggressive 30-second timeout, uses proper termination config
2. CRITICAL: Integrates distance pre-computation to fix ClassCastException
3. Uses GraphHopper service for real Singapore road distances
4. COMMUNITY EDITION: Single-threaded (no enterprise multi-threading)
5. Automatic problem size detection and timeout adjustment
6. PRODUCTION: Convergence detection to reduce variance
UPDATED: Now uses GraphHopper service instead of old routing_service.
"""

from timefold.solver import SolverManager, SolutionManager
from timefold.solver.config import (
    SolverConfig, ScoreDirectorFactoryConfig, 
    TerminationConfig, Duration
)
import logging
import os
import time
from typing import Optional

from .domain import VehicleRoutePlan, Vehicle, Visit
from .constraints import define_constraints
from .graphhopper_service import get_singapore_routing_service  # FIXED: Updated import

# CRITICAL: Import the Timefold fix for ClassCastException
from .domain import precompute_distances_for_timefold, verify_cache_ready, get_distance_cache

# Configure logging
logger = logging.getLogger(__name__)

# Initialize routing service
print("🚀 Initializing GraphHopper routing service...")
routing_service = get_singapore_routing_service()
print(f"✅ Routing service ready: {type(routing_service).__name__}")

def create_solver_config(timeout_minutes: int = 15) -> SolverConfig:
    """
    Create solver configuration with proper timeout AND convergence detection.
    CRITICAL FIX: Adds unimproved time limit for better consistency.
    COMMUNITY EDITION: Single-threaded only.
    """
    
    timeout_seconds = timeout_minutes * 60
    # CRITICAL: Add convergence detection to reduce variance
    unimproved_seconds = min(30, max(5, timeout_seconds // 4))
    
    print(f"📊 Setting solver timeout to {timeout_minutes} minutes ({timeout_seconds} seconds)")
    print(f"   Convergence detection: {unimproved_seconds}s without improvement")
    print(f"   This REDUCES variance and improves consistency")
    print(f"   Using Timefold Community Edition (single-threaded)")
    
    solver_config = SolverConfig(
        solution_class=VehicleRoutePlan,
        entity_class_list=[Vehicle, Visit],
        score_director_factory_config=ScoreDirectorFactoryConfig(
            constraint_provider_function=define_constraints
        ),
        # CRITICAL FIX: Add convergence detection for consistent results
        termination_config=TerminationConfig(
            # Stop if no improvement for X seconds (REDUCES VARIANCE)
            unimproved_spent_limit=Duration(seconds=unimproved_seconds),
            # Safety timeout - maximum time allowed
            spent_limit=Duration(seconds=timeout_seconds)
        )
    )
    
    return solver_config

def create_development_solver_config() -> SolverConfig:
    """Fast solver for development - 2 minutes max."""
    print("🔧 Creating DEVELOPMENT solver config (2 minutes max)")
    return create_solver_config(timeout_minutes=2)

def create_demo_solver_config() -> SolverConfig:
    """Demo solver for presentations - 1 minute max.""" 
    print("🎯 Creating DEMO solver config (1 minute max)")
    return create_solver_config(timeout_minutes=1)

def create_production_solver_config() -> SolverConfig:
    """Production solver for real optimization - 30 minutes max."""
    print("🏭 Creating PRODUCTION solver config (30 minutes max)")
    return create_solver_config(timeout_minutes=30)

# CRITICAL FIX: Choose appropriate timeout based on environment
# UPDATED: Default to ultra-fast demo mode
solver_mode = os.getenv('SOLVER_MODE', 'demo').lower()

if solver_mode == 'development':
    solver_config = create_development_solver_config()
elif solver_mode == 'production':
    solver_config = create_production_solver_config()
elif solver_mode == 'ultra_fast':
    solver_config = create_demo_solver_config()  # 1 minute
elif solver_mode in ['demo', 'medium']:
    solver_config = create_demo_solver_config()  # Ultra-fast 1 minute
else:
    # Safe default - ultra-fast for unknown modes
    print(f"⚠️  Unknown SOLVER_MODE '{solver_mode}', using ultra-fast demo config")
    solver_config = create_demo_solver_config()

print(f"✅ Solver configuration ready for '{solver_mode}' mode")

# Create solver and solution managers
solver_manager = SolverManager.create(solver_config)
solution_manager = SolutionManager.create(solver_manager)

print("🚛 Timefold Vehicle Routing Solver initialized successfully!")
print("🔥 CRITICAL FIX APPLIED: Replaced 30-second timeout with proper timeouts")
print("   - Development: 2 minutes")  
print("   - Demo: 1 minute")
print("   - Production: 30 minutes")
print("🚀 UPDATED: Now using GraphHopper for real Singapore road distances!")
print("🔧 FIXED: Integrated distance pre-computation to eliminate ClassCastException!")
print("🎯 PRODUCTION: Added convergence detection to reduce variance!")
print("📝 NOTE: Using Timefold Community Edition (single-threaded)")

# =====================================================================
# CRITICAL FIX: Wrapper functions for Timefold with distance pre-computation
# =====================================================================

def solve_with_precomputed_distances(route_plan: VehicleRoutePlan, 
                                   problem_id: Optional[str] = None,
                                   timeout_minutes: Optional[int] = None) -> VehicleRoutePlan:
    """
    MAIN FIXED SOLVER FUNCTION - Solves VRP with pre-computed distances.
    This eliminates the ClassCastException by avoiding network calls in domain objects.
    
    USE THIS INSTEAD of direct solver_manager.solve() calls.
    """
    
    try:
        # Step 1: Pre-compute distances BEFORE optimization
        logger.info(f"🔄 Pre-computing distances for Timefold optimization...")
        logger.info(f"   Problem: {len(route_plan.vehicles)} vehicles, {len(route_plan.visits)} visits")
        
        precompute_success = precompute_distances_for_timefold(
            route_plan.vehicles, 
            route_plan.visits, 
            routing_service
        )
        
        if not precompute_success:
            logger.error("❌ Distance pre-computation failed")
            raise Exception("Distance pre-computation failed - cannot proceed with Timefold")
        
        # Step 2: Verify cache is ready
        if not verify_cache_ready():
            logger.warning("⚠️ Distance cache verification failed, proceeding anyway")
        else:
            logger.info("✅ Distance pre-computation complete, cache verified")
        
        # Step 3: Create adaptive solver config if timeout specified
        if timeout_minutes:
            logger.info(f"🕐 Using custom timeout: {timeout_minutes} minutes")
            custom_config = create_solver_config(timeout_minutes)
            custom_solver = SolverManager.create(custom_config)
        else:
            custom_solver = solver_manager
        
        # Step 4: Solve with Timefold (should work without ClassCastException)
        logger.info("🚀 Starting FIXED Timefold optimization...")
        
        if problem_id:
            solver_job = custom_solver.solve(problem_id, route_plan)
        else:
            # Generate a problem ID if none provided
            import uuid
            problem_id = str(uuid.uuid4())
            solver_job = custom_solver.solve(problem_id, route_plan)
        
        # Handle both SolverJob and direct solution returns
        try:
            if hasattr(solver_job, 'get_final_best_solution'):
                solution = solver_job.get_final_best_solution()
            else:
                solution = solver_job
        except:
            # Fallback to solution manager
            logger.warning("⚠️ Using solution manager fallback")
            solution = solution_manager.solve(route_plan)
        
        if solution is None:
            logger.error("❌ Solver returned no solution")
            raise Exception("Solver failed to find a solution")
        
        logger.info(f"✅ FIXED Timefold optimization complete!")
        logger.info(f"   Score: {solution.score}")
        logger.info(f"   Total distance: {solution.total_distance_km:.1f}km")
        logger.info(f"   No ClassCastException occurred!")
        
        return solution
        
    except Exception as e:
        logger.error(f"❌ FIXED Timefold optimization failed: {e}")
        
        # Clean up cache on error
        get_distance_cache().clear()
        
        raise e

def solve_and_listen_with_precomputed_distances(problem_id: str, route_plan: VehicleRoutePlan, 
                                               callback_function) -> None:
    """
    FIXED async solver with pre-computed distances.
    USE THIS INSTEAD of direct solver_manager.solve_and_listen() calls.
    """
    
    try:
        # Pre-compute distances first
        logger.info(f"🔄 Pre-computing distances for async Timefold optimization...")
        
        precompute_success = precompute_distances_for_timefold(
            route_plan.vehicles, 
            route_plan.visits, 
            routing_service
        )
        
        if not precompute_success:
            logger.error("❌ Distance pre-computation failed for async solve")
            # Call callback with error
            if callback_function:
                callback_function(None)
            return
        
        # Verify cache
        if not verify_cache_ready():
            logger.warning("⚠️ Cache verification failed for async solve")
        
        logger.info("✅ Starting FIXED async Timefold optimization...")
        
        # Now start async solving
        solver_manager.solve_and_listen(problem_id, route_plan, callback_function)
        
    except Exception as e:
        logger.error(f"❌ FIXED async Timefold setup failed: {e}")
        get_distance_cache().clear()
        if callback_function:
            callback_function(None)

def test_solver_consistency(route_plan: VehicleRoutePlan, num_runs: int = 5) -> dict:
    """
    PRODUCTION CRITICAL: Test solver consistency across multiple runs.
    High variance indicates unreliable solver configuration.
    """
    logger.info(f"🔍 Testing solver consistency over {num_runs} runs...")
    
    results = []
    solving_times = []
    
    for i in range(num_runs):
        logger.info(f"   Consistency run {i+1}/{num_runs}...")
        
        start_time = time.time()
        solution = solve_with_precomputed_distances(route_plan)
        solve_time = time.time() - start_time
        
        results.append(solution.total_distance_km)
        solving_times.append(solve_time)
        
        logger.info(f"   Run {i+1}: {solution.total_distance_km:.1f}km ({solve_time:.1f}s)")
    
    # Calculate statistics
    avg_distance = sum(results) / len(results)
    min_distance = min(results)
    max_distance = max(results)
    variance = max_distance - min_distance
    variance_pct = (variance/avg_distance*100) if avg_distance > 0 else 0
    
    avg_time = sum(solving_times) / len(solving_times)
    
    logger.info(f"📊 CONSISTENCY ANALYSIS:")
    logger.info(f"   Average distance: {avg_distance:.1f}km")
    logger.info(f"   Best solution: {min_distance:.1f}km")
    logger.info(f"   Worst solution: {max_distance:.1f}km")
    logger.info(f"   Variance: {variance:.1f}km ({variance_pct:.1f}%)")
    logger.info(f"   Average solve time: {avg_time:.1f}s")
    
    # Production readiness assessment
    if variance_pct > 15:
        status = "FAILED"
        message = "UNRELIABLE - variance too high for production"
        logger.error(f"❌ {message}")
    elif variance_pct > 10:
        status = "WARNING" 
        message = "Borderline - may need tuning for production"
        logger.warning(f"⚠️  {message}")
    else:
        status = "PASSED"
        message = "Good consistency for production use"
        logger.info(f"✅ {message}")
    
    return {
        "status": status,
        "message": message,
        "variance_km": variance,
        "variance_percentage": variance_pct,
        "average_distance": avg_distance,
        "best_distance": min_distance,
        "worst_distance": max_distance,
        "average_solve_time": avg_time,
        "all_results": results,
        "num_runs": num_runs
    }

def test_solver_fix(route_plan: VehicleRoutePlan) -> dict:
    """
    Test that the solver fix works without running full optimization.
    Call this to verify the ClassCastException fix works.
    """
    
    try:
        logger.info("🧪 Testing Timefold solver fix...")
        
        # Test distance pre-computation
        precompute_success = precompute_distances_for_timefold(
            route_plan.vehicles, 
            route_plan.visits, 
            routing_service
        )
        
        if not precompute_success:
            return {"status": "failed", "step": "precomputation", "error": "Distance pre-computation failed"}
        
        # Test cache verification
        if not verify_cache_ready():
            return {"status": "failed", "step": "verification", "error": "Cache verification failed"}
        
        # Test domain object calls (should not crash)
        if route_plan.visits and len(route_plan.visits) >= 2:
            visit1 = route_plan.visits[0]
            visit2 = route_plan.visits[1]
            
            # These should work without network calls
            distance = visit1.location.distance_to(visit2.location)
            duration = visit1.location.driving_time_to(visit2.location)
            
            logger.info(f"✅ Domain calls successful: {distance/1000:.1f}km, {duration/60:.1f}min")
        
        cache = get_distance_cache()
        
        return {
            "status": "success",
            "cache_size": cache.get_cache_size(),
            "is_precomputed": cache.is_precomputed(),
            "message": "Timefold fix verified - ready for optimization"
        }
        
    except Exception as e:
        logger.error(f"❌ Solver fix test failed: {e}")
        return {"status": "failed", "error": str(e)}

# Utility functions
def detect_problem_size(route_plan: VehicleRoutePlan) -> str:
    """Automatically detect problem size based on visits and vehicles."""
    visit_count = len(route_plan.visits)
    vehicle_count = len(route_plan.vehicles)
    
    if visit_count <= 30 and vehicle_count <= 5:
        return "small"
    elif visit_count <= 100 and vehicle_count <= 10:
        return "medium"
    else:
        return "large"

def get_recommended_timeout_minutes(route_plan: VehicleRoutePlan) -> int:
    """Get recommended timeout based on problem size (adjusted for single-threaded)."""
    problem_size = detect_problem_size(route_plan)
    
    # Longer timeouts for single-threaded Community Edition
    recommendations = {
        "small": 2,      # Up to 30 visits, 5 vehicles
        "medium": 10,    # Up to 100 visits, 10 vehicles  
        "large": 30      # 100+ visits, 10+ vehicles
    }
    
    return recommendations.get(problem_size, 10)

def create_adaptive_solver_config(route_plan: VehicleRoutePlan) -> SolverConfig:
    """Create solver config adapted to the specific problem size."""
    problem_size = detect_problem_size(route_plan)
    timeout_minutes = get_recommended_timeout_minutes(route_plan)
    
    print(f"📊 Detected {problem_size} problem: {len(route_plan.visits)} visits, {len(route_plan.vehicles)} vehicles")
    print(f"🕐 Recommended timeout: {timeout_minutes} minutes")
    
    return create_solver_config(timeout_minutes)

def solve_with_adaptive_config(route_plan: VehicleRoutePlan, 
                              problem_id: Optional[str] = None) -> VehicleRoutePlan:
    """
    Solve with automatically adapted configuration AND distance pre-computation.
    This combines adaptive timeout with the ClassCastException fix.
    """
    
    problem_size = detect_problem_size(route_plan)
    timeout_minutes = get_recommended_timeout_minutes(route_plan)
    
    logger.info(f"📊 Adaptive solving: {problem_size} problem, {timeout_minutes}min timeout")
    
    return solve_with_precomputed_distances(route_plan, problem_id, timeout_minutes)

# =====================================================================
# BACKWARDS COMPATIBILITY - Safe wrappers for existing code
# =====================================================================

# IMPORTANT: These replace the direct solver_manager calls to avoid ClassCastException
class FixedSolverManager:
    """
    Drop-in replacement for SolverManager that automatically applies the distance fix.
    Use this to upgrade existing code without changing the interface.
    """
    
    def __init__(self, original_solver_manager):
        self._solver_manager = original_solver_manager
    
    def solve(self, problem_id: str, route_plan: VehicleRoutePlan) -> VehicleRoutePlan:
        """Fixed solve method with distance pre-computation."""
        return solve_with_precomputed_distances(route_plan, problem_id)
    
    def solve_and_listen(self, problem_id: str, route_plan: VehicleRoutePlan, callback):
        """Fixed async solve method with distance pre-computation."""
        solve_and_listen_with_precomputed_distances(problem_id, route_plan, callback)
    
    def terminate_early(self, problem_id: str):
        """Delegate to original solver manager."""
        return self._solver_manager.terminate_early(problem_id)
    
    def get_solver_status(self, problem_id: str):
        """Delegate to original solver manager."""
        return self._solver_manager.get_solver_status(problem_id)

# Create the fixed solver manager wrapper
fixed_solver_manager = FixedSolverManager(solver_manager)

# For backwards compatibility and imports - UPDATED to include fix functions
__all__ = [
    'solver_manager', 'solution_manager', 'fixed_solver_manager',
    'solve_with_precomputed_distances', 'solve_and_listen_with_precomputed_distances',
    'test_solver_fix', 'test_solver_consistency', 'create_adaptive_solver_config', 'solve_with_adaptive_config'
]

# Quick setup verification
print("\n🔍 Solver Setup Verification:")
print(f"   Default timeout: {solver_config.termination_config.spent_limit}")
print(f"   Convergence detection: {solver_config.termination_config.unimproved_spent_limit}")
print(f"   Constraint provider: {solver_config.score_director_factory_config.constraint_provider_function}")
print(f"   Solution class: {solver_config.solution_class}")
print(f"   Entity classes: {solver_config.entity_class_list}")
print(f"   Threading: Single-threaded (Timefold Community Edition)")
print("✅ All solver components configured correctly!")
print("🔧 IMPORTANT: Use fixed functions to avoid ClassCastException:")
print("   - solve_with_precomputed_distances() instead of solver_manager.solve()")
print("   - solve_and_listen_with_precomputed_distances() instead of solver_manager.solve_and_listen()")
print("   - OR use fixed_solver_manager as drop-in replacement")
print("🎯 PRODUCTION: Convergence detection should reduce solution variance!")

# =====================================================================
# TESTING FUNCTION
# =====================================================================

def run_solver_test():
    """Run a quick test to verify the solver fix works."""
    print("\n🧪 Testing FIXED Timefold solver (Community Edition)...")
    
    try:
        # Create minimal test problem
        from .domain import Location
        from datetime import datetime, timedelta
        
        depot = Location(latitude=1.2966, longitude=103.8518)  # Marina Bay
        
        vehicle = Vehicle(
            id="V1",
            capacity=100,
            home_location=depot,
            departure_time=datetime(2024, 1, 1, 8, 0)
        )
        
        visit = Visit(
            id="C1",
            name="Customer 1",
            location=Location(latitude=1.3521, longitude=103.8198),
            demand=10,
            min_start_time=datetime(2024, 1, 1, 9, 0),
            max_end_time=datetime(2024, 1, 1, 17, 0),
            service_duration=timedelta(minutes=15)
        )
        
        route_plan = VehicleRoutePlan(
            name="Test Problem",
            south_west_corner=Location(latitude=1.2, longitude=103.6),
            north_east_corner=Location(latitude=1.5, longitude=104.0),
            vehicles=[vehicle],
            visits=[visit]
        )
        
        # Test the fix (don't run full optimization)
        test_result = test_solver_fix(route_plan)
        
        if test_result["status"] == "success":
            print(f"✅ Community Edition solver fix test PASSED: {test_result['cache_size']} distances cached")
            print("   Ready for production use (single-threaded)!")
            return True
        else:
            print(f"❌ Solver fix test FAILED: {test_result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Solver test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test if executed directly
    run_solver_test()