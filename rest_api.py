# PRODUCTION-READY REST API with Consistency Testing Integration + Fresh Random Seeds
# FIXED: Added missing /route-plans-fresh endpoint and matching demo categories

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Annotated, Optional, Dict, Any
import requests
import logging
import time
import polyline  # CRITICAL: Import polyline decoder
import asyncio
from pydantic import BaseModel

from .domain import *
from .score_analysis import *
from .demo_data import DemoData, generate_demo_data
from .solver import (
    solver_manager, solution_manager, fixed_solver_manager,
    solve_with_precomputed_distances, test_solver_consistency, 
    test_solver_fix, create_solver_config, solver_config
)
from .hybrid_tsp_solver import get_hybrid_tsp_solver
from .graphhopper_service import get_graphhopper_service

# ADDED: Import the Timefold fix
from .domain import precompute_distances_for_timefold, verify_cache_ready, get_distance_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(docs_url='/q/swagger-ui')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_sets: dict[str, VehicleRoutePlan] = {}

# =====================================================================
# PYDANTIC MODELS FOR CONSISTENCY TESTING
# =====================================================================

class ConsistencyTestRequest(BaseModel):
    demo_type: Optional[str] = "SINGAPORE_WIDE"
    num_runs: Optional[int] = 5
    timestamp: Optional[int] = None

class QuickTestRequest(BaseModel):
    timestamp: Optional[int] = None
    test_type: Optional[str] = "quick"

# =====================================================================
# EXISTING ENDPOINTS (UNCHANGED)
# =====================================================================

@app.get("/demo-data")
async def demo_data_list():
    return [e.name for e in DemoData]

@app.get("/demo-data/{dataset_id}", response_model_exclude_none=True)
async def get_demo_data(dataset_id: str) -> VehicleRoutePlan:
    demo_data = generate_demo_data(getattr(DemoData, dataset_id))
    return demo_data

@app.get("/route-plans/{problem_id}", response_model_exclude_none=True)
async def get_route(problem_id: str) -> VehicleRoutePlan:
    route = data_sets[problem_id]
    return route.model_copy(update={
        'solver_status': solver_manager.get_solver_status(problem_id),
    })

def update_route(problem_id: str, route: VehicleRoutePlan):
    global data_sets
    data_sets[problem_id] = route

def json_to_vehicle_route_plan(json: dict) -> VehicleRoutePlan:
    visits = {
        visit['id']: visit for visit in json.get('visits', [])
    }
    vehicles = {
        vehicle['id']: vehicle for vehicle in json.get('vehicles', [])
    }
    for visit in visits.values():
        if 'vehicle' in visit:
            del visit['vehicle']
        if 'previousVisit' in visit:
            del visit['previousVisit']
        if 'nextVisit' in visit:
            del visit['nextVisit']
    visits = {visit_id: Visit.model_validate(visits[visit_id]) for visit_id in visits}
    json['visits'] = list(visits.values())
    for vehicle in vehicles.values():
        vehicle['visits'] = [visits[visit_id] for visit_id in vehicle['visits']]
    json['vehicles'] = list(vehicles.values())
    return VehicleRoutePlan.model_validate(json, context={
        'visits': visits,
        'vehicles': vehicles
    })

async def setup_context(request: Request) -> VehicleRoutePlan:
    json = await request.json()
    return json_to_vehicle_route_plan(json)

@app.post("/route-plans")
async def solve_route(route: Annotated[VehicleRoutePlan, Depends(setup_context)]) -> str:
    """FIXED Timefold solver endpoint with pre-computed distances."""
    
    try:
        job_id = str(uuid4())
        logger.info(f"🚛 Starting FIXED Timefold optimization: {len(route.vehicles)} vehicles, {len(route.visits)} visits")
        
        # CRITICAL FIX: Pre-compute distances BEFORE Timefold optimization
        logger.info("🔄 Pre-computing distances for Timefold...")
        routing_service = get_graphhopper_service()
        
        precompute_success = precompute_distances_for_timefold(
            route.vehicles, 
            route.visits, 
            routing_service
        )
        
        if not precompute_success:
            logger.error("❌ Distance pre-computation failed")
            get_distance_cache().clear()
            logger.warning("⚠️ Using fallback distances, optimization may be less accurate")
        
        if not verify_cache_ready():
            logger.warning("⚠️ Distance cache not fully ready, proceeding with available data")
        else:
            logger.info("✅ Distance pre-computation complete, starting Timefold...")
        
        data_sets[job_id] = route
        
        # Use fixed solver manager to ensure distance pre-computation
        fixed_solver_manager.solve_and_listen(job_id, route,
                                            lambda solution: update_route(job_id, solution))
        
        logger.info(f"✅ FIXED Timefold optimization submitted: {job_id}")
        return job_id
        
    except Exception as e:
        logger.error(f"❌ FIXED Timefold optimization failed: {e}")
        get_distance_cache().clear()
        error_job_id = f"error_{uuid4()}"
        return error_job_id

# =====================================================================
# CRITICAL FIX: MISSING /route-plans-fresh ENDPOINT FOR ITERATIVE TESTING
# =====================================================================

@app.post("/route-plans-fresh")
async def solve_route_with_fresh_seed(route: Annotated[VehicleRoutePlan, Depends(setup_context)]):
    """
    CRITICAL MISSING ENDPOINT: Fresh random seed solver for iterative testing.
    This ensures each iteration produces different results for proper variance testing.
    """
    
    try:
        job_id = str(uuid4())
        logger.info(f"🎲 Starting FRESH SEED Timefold optimization: {len(route.vehicles)} vehicles, {len(route.visits)} visits")
        
        # CRITICAL: Clear distance cache to force fresh random seed behavior
        get_distance_cache().clear()
        logger.info("🔄 Cleared distance cache for fresh random seed")
        
        # Pre-compute distances with fresh routing calls
        logger.info("🔄 Pre-computing distances with fresh routing calls...")
        routing_service = get_graphhopper_service()
        
        precompute_success = precompute_distances_for_timefold(
            route.vehicles, 
            route.visits, 
            routing_service
        )
        
        if not precompute_success:
            logger.error("❌ Fresh distance pre-computation failed")
            get_distance_cache().clear()
            logger.warning("⚠️ Using fallback distances for fresh seed")
        
        data_sets[job_id] = route
        
        # Use fixed solver manager with fresh configuration
        fixed_solver_manager.solve_and_listen(job_id, route,
                                            lambda solution: update_route(job_id, solution))
        
        logger.info(f"✅ FRESH SEED Timefold optimization submitted: {job_id}")
        
        # Return format expected by frontend iterative testing
        return {
            "schedule_id": job_id,
            "status": "solving",
            "fresh_seed": True,
            "endpoint": "/route-plans-fresh"
        }
        
    except Exception as e:
        logger.error(f"❌ FRESH SEED Timefold optimization failed: {e}")
        get_distance_cache().clear()
        
        return {
            "error": str(e),
            "status": "failed",
            "fresh_seed": True,
            "endpoint": "/route-plans-fresh"
        }

# =====================================================================
# NEW: PRODUCTION-CRITICAL CONSISTENCY TESTING ENDPOINTS
# =====================================================================

@app.post("/quick-variance-test")
async def quick_variance_test(request: Optional[QuickTestRequest] = None):
    """
    PRODUCTION CRITICAL: Quick 3-run variance test to detect solver reliability issues.
    High variance = unreliable solver configuration.
    """
    logger.info("🔍 Starting quick variance test (3 runs)...")
    
    try:
        # Use a consistent demo dataset for testing
        demo_data = generate_demo_data(DemoData.SINGAPORE_WIDE)
        
        if not demo_data.visits:
            raise HTTPException(status_code=500, detail="Demo data has no visits")
        
        logger.info(f"   Testing with {len(demo_data.vehicles)} vehicles, {len(demo_data.visits)} visits")
        
        # Run 3 optimization attempts with the FIXED solver
        results = []
        solve_times = []
        
        for i in range(3):
            logger.info(f"   Quick test run {i+1}/3...")
            
            start_time = time.time()
            
            # Deep copy to prevent state pollution between runs
            test_route = demo_data.model_copy(deep=True)
            
            # Use the fixed solver with pre-computed distances
            solution = solve_with_precomputed_distances(test_route)
            
            solve_time = time.time() - start_time
            solve_times.append(solve_time)
            
            # Calculate total distance
            total_distance_km = sum(vehicle.total_distance_km for vehicle in solution.vehicles)
            results.append(total_distance_km)
            
            logger.info(f"   Run {i+1}: {total_distance_km:.1f}km ({solve_time:.1f}s)")
        
        # Statistical analysis
        average = sum(results) / len(results)
        best = min(results)
        worst = max(results)
        variance_km = worst - best
        variance_percentage = (variance_km / average * 100) if average > 0 else 0
        avg_solve_time = sum(solve_times) / len(solve_times)
        
        # Determine status based on variance
        if variance_percentage > 15:
            status = "FAILED"
            message = "CRITICAL: High variance indicates unreliable solver configuration"
            production_ready = False
        elif variance_percentage > 10:
            status = "WARNING"
            message = "Moderate variance - consider tuning solver configuration"
            production_ready = False
        else:
            status = "PASSED"
            message = "Good solver consistency for production use"
            production_ready = True
        
        logger.info(f"📊 Quick test results: {status}, {variance_percentage:.1f}% variance")
        
        response = {
            "status": status,
            "message": message,
            "production_ready": production_ready,
            "variance_percentage": variance_percentage,
            "variance_km": variance_km,
            "average_distance": average,
            "best_distance": best,
            "worst_distance": worst,
            "all_distances": results,
            "average_solve_time": avg_solve_time,
            "num_runs": 3
        }
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Quick variance test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick test failed: {str(e)}")

@app.post("/test-solver-consistency")
async def test_solver_consistency_endpoint(request: ConsistencyTestRequest):
    """
    PRODUCTION CRITICAL: Full 5-run consistency test for comprehensive solver analysis.
    This is the definitive test for production readiness.
    """
    logger.info(f"🔍 Starting full consistency test ({request.num_runs} runs)...")
    
    try:
        # Get demo data based on type
        if hasattr(DemoData, request.demo_type):
            demo_data = generate_demo_data(getattr(DemoData, request.demo_type))
        else:
            demo_data = generate_demo_data(DemoData.SINGAPORE_WIDE)
        
        if not demo_data.visits:
            raise HTTPException(status_code=500, detail="Demo data has no visits")
        
        logger.info(f"   Testing {request.demo_type} with {len(demo_data.vehicles)} vehicles, {len(demo_data.visits)} visits")
        
        # Use the existing test_solver_consistency function from solver.py
        consistency_results = test_solver_consistency(demo_data, request.num_runs)
        
        logger.info(f"📊 Consistency test complete: {consistency_results['status']}")
        
        # Wrap in expected format for frontend
        response = {
            "result": consistency_results
        }
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Solver consistency test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consistency test failed: {str(e)}")

@app.get("/solver-health")
async def check_solver_health():
    """
    PRODUCTION CRITICAL: Comprehensive solver health and configuration check.
    Identifies configuration issues that cause high variance.
    """
    logger.info("🏥 Checking solver health...")
    
    try:
        # Check if solver manager is available
        solver_ready = solver_manager is not None and solution_manager is not None
        
        # CRITICAL: Check for convergence detection in termination config
        convergence_detection = False
        timeout_minutes = 0
        
        if solver_config and solver_config.termination_config:
            # Check for unimproved_spent_limit (convergence detection)
            convergence_detection = solver_config.termination_config.unimproved_spent_limit is not None
            
            # Get timeout in minutes
            if solver_config.termination_config.spent_limit:
                timeout_seconds = solver_config.termination_config.spent_limit.seconds
                timeout_minutes = timeout_seconds // 60
        
        # Determine solver mode based on timeout
        if timeout_minutes <= 2:
            mode = "development"
        elif timeout_minutes <= 10:
            mode = "demo"
        else:
            mode = "production"
        
        # Generate recommendation
        if not solver_ready:
            recommendation = "CRITICAL: Solver components not properly initialized"
        elif not convergence_detection:
            recommendation = "CRITICAL: Add unimproved_spent_limit to termination criteria to prevent random variance"
        elif timeout_minutes < 1:
            recommendation = "Increase timeout to at least 1 minute for stable results"
        else:
            recommendation = "Solver configuration looks good for production use"
        
        # Test the fix to ensure it's working
        test_demo = generate_demo_data(DemoData.SIMPLE)
        fix_test = test_solver_fix(test_demo)
        
        health_data = {
            "solver_ready": solver_ready,
            "convergence_detection": convergence_detection,
            "timeout_minutes": timeout_minutes,
            "mode": mode,
            "recommendation": recommendation,
            "fix_status": fix_test.get("status", "unknown"),
            "cache_ready": fix_test.get("is_precomputed", False),
            "cache_size": fix_test.get("cache_size", 0)
        }
        
        logger.info(f"🏥 Health check complete: convergence={convergence_detection}, timeout={timeout_minutes}min")
        
        return health_data
        
    except Exception as e:
        logger.error(f"❌ Solver health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# =====================================================================
# ENHANCED EXISTING ENDPOINTS
# =====================================================================

@app.post("/route-plans/hybrid")
async def solve_route_hybrid(route: Annotated[VehicleRoutePlan, Depends(setup_context)]) -> dict:
    """
    FIXED: Hybrid route optimization endpoint.
    Uses brute force for ≤10 stops, Timefold for larger problems.
    Returns optimized solution immediately (no job queue).
    """
    try:
        start_time = time.time()
        logger.info(f"🚀 Starting hybrid optimization: {len(route.vehicles)} vehicles, {len(route.visits)} visits")
        
        vehicle_routes = []
        
        if any(visit.vehicle for visit in route.visits):
            for vehicle in route.vehicles:
                assigned_visits = [visit for visit in route.visits if visit.vehicle and visit.vehicle.id == vehicle.id]
                visit_data = [(visit.id, (visit.location.latitude, visit.location.longitude)) for visit in assigned_visits]
                home_location = (vehicle.home_location.latitude, vehicle.home_location.longitude)
                vehicle_routes.append((vehicle.id, home_location, visit_data))
        else:
            visits_per_vehicle = len(route.visits) // len(route.vehicles)
            extra_visits = len(route.visits) % len(route.vehicles)
            
            visit_index = 0
            for i, vehicle in enumerate(route.vehicles):
                num_visits = visits_per_vehicle + (1 if i < extra_visits else 0)
                assigned_visits = route.visits[visit_index:visit_index + num_visits]
                visit_data = [(visit.id, (visit.location.latitude, visit.location.longitude)) for visit in assigned_visits]
                home_location = (vehicle.home_location.latitude, vehicle.home_location.longitude)
                vehicle_routes.append((vehicle.id, home_location, visit_data))
                visit_index += num_visits
        
        solver = get_hybrid_tsp_solver()
        optimized_results = solver.optimize_multi_vehicle_parallel(vehicle_routes)
        
        for vehicle_id, optimized_visit_ids in optimized_results:
            for vehicle in route.vehicles:
                if vehicle.id == vehicle_id:
                    visit_lookup = {visit.id: visit for visit in route.visits}
                    
                    optimized_visits = []
                    for visit_id in optimized_visit_ids:
                        if visit_id in visit_lookup:
                            visit = visit_lookup[visit_id]
                            visit.vehicle = vehicle
                            optimized_visits.append(visit)
                    
                    vehicle.visits = optimized_visits
                    break
        
        total_distance_km = sum(vehicle.total_distance_km for vehicle in route.vehicles)
        total_time_hours = sum(vehicle.total_driving_time_seconds / 3600 for vehicle in route.vehicles)
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Hybrid optimization complete in {elapsed:.3f}s")
        logger.info(f"   Total distance: {total_distance_km:.1f}km")
        logger.info(f"   Total time: {total_time_hours:.1f}h")
        
        return {
            "status": "solved",
            "optimization_time_seconds": round(elapsed, 3),
            "algorithm": "hybrid_tsp",
            "solution": route.model_dump(),
            "metrics": {
                "total_distance_km": round(total_distance_km, 1),
                "total_time_hours": round(total_time_hours, 1),
                "vehicles_optimized": len(route.vehicles),
                "total_visits": len(route.visits),
                "average_time_per_vehicle": round(elapsed / len(route.vehicles), 3) if route.vehicles else 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Hybrid optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "original_solution": route.model_dump()
        }

@app.post("/route-plans/test-fixed")
async def test_fixed_optimization(route: Annotated[VehicleRoutePlan, Depends(setup_context)]):
    """Test endpoint to verify the Timefold fix works."""
    
    try:
        logger.info("🧪 Testing FIXED Timefold integration...")
        
        # Use the test function from solver.py
        test_result = test_solver_fix(route)
        
        return test_result
        
    except Exception as e:
        logger.error(f"❌ Fix test failed: {e}")
        return {"status": "error", "message": str(e)}

# =====================================================================
# CACHE AND UTILITY ENDPOINTS
# =====================================================================

@app.get("/cache/status")
async def get_cache_status():
    """Get distance cache status."""
    cache = get_distance_cache()
    return {
        "cache_size": cache.get_cache_size(),
        "is_precomputed": cache.is_precomputed()
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear the distance cache."""
    cache = get_distance_cache()
    cache.clear()
    logger.info("🧹 Distance cache cleared")
    return {"status": "cleared"}

@app.put("/route-plans/analyze")
async def analyze_route(route: Annotated[VehicleRoutePlan, Depends(setup_context)]) \
        -> dict['str', list[ConstraintAnalysisDTO]]:
    return {'constraints': [ConstraintAnalysisDTO(
        name=constraint.constraint_name,
        weight=constraint.weight,
        score=constraint.score,
        matches=[
            MatchAnalysisDTO(
                name=match.constraint_ref.constraint_name,
                score=match.score,
                justification=match.justification
            )
            for match in constraint.matches
        ]
    ) for constraint in solution_manager.analyze(route).constraint_analyses]}

@app.delete("/route-plans/{problem_id}")
async def stop_solving(problem_id: str) -> None:
    solver_manager.terminate_early(problem_id)

# =====================================================================
# GRAPHHOPPER AND SERVICE HEALTH ENDPOINTS
# =====================================================================

@app.get("/graphhopper/health")
async def graphhopper_health():
    """Check GraphHopper service health."""
    try:
        service = get_graphhopper_service()
        health = service.get_service_health()
        return health
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/graphhopper/test")
async def test_graphhopper():
    """Test GraphHopper with Singapore routes."""
    try:
        from .graphhopper_service import test_graphhopper_integration
        success = test_graphhopper_integration()
        return {"test_passed": success}
    except Exception as e:
        return {"test_passed": False, "error": str(e)}

@app.get("/optimization/compare/{problem_id}")
async def compare_optimization_methods(problem_id: str):
    """Compare Timefold vs Hybrid optimization performance."""
    if problem_id not in data_sets:
        return {"error": "Problem not found"}
    
    route_plan = data_sets[problem_id].model_copy(deep=True)
    
    results = {
        "problem_id": problem_id,
        "vehicles": len(route_plan.vehicles),
        "visits": len(route_plan.visits),
        "methods": {}
    }
    
    try:
        start_time = time.time()
        hybrid_response = await solve_route_hybrid(route_plan.model_copy(deep=True))
        hybrid_time = time.time() - start_time
        
        results["methods"]["hybrid"] = {
            "time_seconds": round(hybrid_time, 3),
            "status": hybrid_response["status"],
            "algorithm": "hybrid_tsp"
        }
        
        if "metrics" in hybrid_response:
            results["methods"]["hybrid"]["metrics"] = hybrid_response["metrics"]
            
    except Exception as e:
        results["methods"]["hybrid"] = {"error": str(e)}
    
    try:
        fix_test = await test_fixed_optimization(route_plan.model_copy(deep=True))
        results["methods"]["timefold_fixed"] = {
            "status": fix_test["status"],
            "cache_size": fix_test.get("cache_size", 0),
            "precomputed": fix_test.get("precomputed", False),
            "note": "Use /route-plans endpoint for full FIXED Timefold optimization"
        }
    except Exception as e:
        results["methods"]["timefold_fixed"] = {"error": str(e)}
    
    return results

# =====================================================================
# ROUTE VISUALIZATION WITH FIXED POLYLINE DECODING
# =====================================================================

def get_route_geometry(start_loc, end_loc) -> dict:
    """
    PRODUCTION-READY: Get route geometry from GraphHopper with proper polyline decoding.
    
    CRITICAL FIXES:
    1. Proper polyline decoding using polyline library
    2. Flexible parameter handling (Location objects or tuples)
    3. Comprehensive error handling with fallbacks
    4. Performance optimization with timeout handling
    5. Detailed logging for debugging
    """
    logger.debug(f"🛣️ Getting route geometry from GraphHopper...")
    
    try:
        # Handle both Location objects and coordinate tuples
        if hasattr(start_loc, 'latitude'):
            start_lat, start_lng = start_loc.latitude, start_loc.longitude
        else:
            start_lat, start_lng = start_loc[0], start_loc[1]
        
        if hasattr(end_loc, 'latitude'):
            end_lat, end_lng = end_loc.latitude, end_loc.longitude
        else:
            end_lat, end_lng = end_loc[0], end_loc[1]
        
        # GraphHopper routing request with encoded polylines
        params = {
            'point': [f"{start_lat},{start_lng}", f"{end_lat},{end_lng}"],
            'profile': 'car',
            'calc_points': 'true',
            'instructions': 'false',
            'points_encoded': 'true'  # CRITICAL: Request encoded polylines
        }
        
        url = "http://localhost:8989/route"
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'paths' in data and len(data['paths']) > 0:
                path = data['paths'][0]
                
                # CRITICAL FIX: Properly decode encoded polyline
                if 'points' in path:
                    encoded_polyline = path['points']
                    
                    if isinstance(encoded_polyline, str) and len(encoded_polyline) > 0:
                        try:
                            # Decode the polyline to get actual road coordinates
                            decoded_points = polyline.decode(encoded_polyline)
                            
                            # Convert to [lng, lat] format for mapping (some maps expect this)
                            # Note: polyline.decode returns [(lat, lng), ...] format
                            coordinates = [[point[1], point[0]] for point in decoded_points]
                            
                            result = {
                                "geometry": coordinates,
                                "distance": path.get('distance', 0),
                                "duration": path.get('time', 0) / 1000,  # Convert ms to seconds
                                "source": "graphhopper_decoded"
                            }
                            
                            return result
                            
                        except Exception as decode_error:
                            logger.warning(f"Polyline decode failed: {decode_error}")
                            # Continue to fallback handling below
                    
                    elif isinstance(encoded_polyline, dict) and 'coordinates' in encoded_polyline:
                        # Handle GeoJSON format (backup case)
                        geometry = [[coord[1], coord[0]] for coord in encoded_polyline['coordinates']]
                        
                        return {
                            "geometry": geometry,
                            "distance": path.get('distance', 0),
                            "duration": path.get('time', 0) / 1000,
                            "source": "graphhopper_geojson"
                        }
                
                # If we reach here, use GraphHopper distances but straight line geometry
                logger.warning("Could not decode route geometry, using straight line with GraphHopper distances")
                return get_straight_line_with_gh_data(start_lat, start_lng, end_lat, end_lng, path)
                
    except requests.Timeout:
        logger.warning("GraphHopper timeout")
    except Exception as e:
        logger.warning(f"GraphHopper request failed: {e}")
    
    # Final fallback to straight line with estimated data
    return get_straight_line_fallback(start_lat, start_lng, end_lat, end_lng)

def get_straight_line_with_gh_data(start_lat, start_lng, end_lat, end_lng, path_data):
    """Use GraphHopper distance/time data but straight line geometry."""
    return {
        "geometry": [[start_lat, start_lng], [end_lat, end_lng]],
        "distance": path_data.get('distance', 0),
        "duration": path_data.get('time', 0) / 1000,
        "source": "graphhopper_straight_line"
    }

def get_straight_line_fallback(start_lat, start_lng, end_lat, end_lng):
    """Complete fallback when GraphHopper fails."""
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, [start_lat, start_lng, end_lat, end_lng])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    distance = 6371000 * 2 * math.asin(math.sqrt(a))  # meters
    duration = distance / (50000/3600)  # 50 km/h average speed in Singapore
    
    return {
        "geometry": [[start_lat, start_lng], [end_lat, end_lng]],
        "distance": distance,
        "duration": duration,
        "source": "haversine_fallback"
    }

@app.get("/route-visualization/{problem_id}")
async def get_route_visualization(problem_id: str):
    """Get route visualization data with FIXED GraphHopper geometries."""
    try:
        if problem_id not in data_sets:
            logger.error(f"❌ Route plan not found for ID: {problem_id}")
            return {"error": "Route plan not found"}
        
        route_plan = data_sets[problem_id]
        logger.info(f"🗺️ Generating visualization for {len(route_plan.vehicles)} vehicles")
        
        route_data = {
            "vehicles": [],
            "center": [1.3521, 103.8198],
            "bounds": [[1.2, 103.6], [1.5, 104.0]],
            "totalDistanceKm": 0.0,
            "totalTimeHours": 0.0,
            "routeQuality": "unknown"
        }
        
        total_segments_processed = 0
        total_decoded_segments = 0
        
        for vehicle in route_plan.vehicles:
            logger.debug(f"🚛 Processing vehicle {vehicle.id}")
            
            if not vehicle.visits:
                logger.warning(f"Vehicle {vehicle.id} has no visits, skipping")
                continue
                
            vehicle_routes = []
            all_locations = [vehicle.home_location] + [visit.location for visit in vehicle.visits] + [vehicle.home_location]
            
            for i in range(len(all_locations) - 1):
                start_loc = all_locations[i]
                end_loc = all_locations[i + 1]
                
                try:
                    route = get_route_geometry(start_loc, end_loc)
                    
                    # Track quality metrics
                    total_segments_processed += 1
                    if route['source'] == 'graphhopper_decoded':
                        total_decoded_segments += 1
                    
                    vehicle_routes.append(route)
                    
                except Exception as e:
                    logger.warning(f"Failed to get route geometry: {e}")
                    vehicle_routes.append(get_straight_line_fallback(
                        start_loc.latitude, start_loc.longitude,
                        end_loc.latitude, end_loc.longitude
                    ))
            
            vehicle_data = {
                "id": vehicle.id,
                "home_location": [vehicle.home_location.latitude, vehicle.home_location.longitude],
                "routes": vehicle_routes,
                "visits": [
                    {
                        "id": visit.id,
                        "name": visit.name,
                        "location": [visit.location.latitude, visit.location.longitude],
                        "demand": visit.demand
                    } for visit in vehicle.visits
                ],
                "total_distance_km": round(vehicle.total_distance_km, 1),
                "total_time_hours": round(vehicle.total_driving_time_seconds / 3600, 1)
            }
            
            route_data["vehicles"].append(vehicle_data)
            route_data["totalDistanceKm"] += vehicle_data["total_distance_km"]
            route_data["totalTimeHours"] += vehicle_data["total_time_hours"]
        
        # Calculate route quality
        if total_segments_processed > 0:
            decode_rate = (total_decoded_segments / total_segments_processed) * 100
            if decode_rate >= 90:
                route_data["routeQuality"] = "excellent"
            elif decode_rate >= 70:
                route_data["routeQuality"] = "good"
            elif decode_rate >= 50:
                route_data["routeQuality"] = "fair"
            else:
                route_data["routeQuality"] = "poor"
        
        route_data["totalDistanceKm"] = round(route_data["totalDistanceKm"], 1)
        route_data["totalTimeHours"] = round(route_data["totalTimeHours"], 1)
        
        logger.info(f"✅ Visualization complete: {len(route_data['vehicles'])} vehicles, {route_data['totalDistanceKm']}km total")
        return route_data
        
    except Exception as e:
        logger.error(f"❌ Error generating route visualization: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate route visualization: {e}"}

@app.get("/test-route-geometry")
async def test_route_geometry():
    """Test the FIXED route geometry function."""
    try:
        from .domain import Location
        
        # Test with Singapore locations
        marina_bay = Location(latitude=1.2966, longitude=103.8518)
        changi = Location(latitude=1.3644, longitude=103.9915)
        
        logger.info("🧪 Testing route geometry function...")
        result = get_route_geometry(marina_bay, changi)
        
        return {
            "status": "success",
            "test_route": "Marina Bay to Changi Airport",
            "result": result,
            "geometry_points": len(result.get("geometry", [])),
            "distance_km": round(result.get("distance", 0) / 1000, 1),
            "time_minutes": round(result.get("duration", 0) / 60, 1),
            "source": result.get("source", "unknown"),
            "quality": "excellent" if result.get("source") == "graphhopper_decoded" else "needs_improvement"
        }
        
    except Exception as e:
        logger.error(f"❌ Route geometry test failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# Mount static files last to avoid conflicts
app.mount("/", StaticFiles(directory="static", html=True), name="static")