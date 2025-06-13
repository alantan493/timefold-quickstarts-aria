"""
Production VRP Constraints
Minimal, fast constraints for commercial deployment.
"""

from timefold.solver.score import ConstraintFactory, HardSoftScore, constraint_provider
from .domain import Vehicle, Visit

@constraint_provider
def define_constraints(factory: ConstraintFactory):
    """Production constraint set - fast and reliable."""
    return [
        # Hard constraints
        vehicle_capacity(factory),
        service_time_windows(factory),
        
        # Soft constraints  
        minimize_unassigned_visits(factory),
        minimize_total_travel_time(factory),
    ]

def vehicle_capacity(factory: ConstraintFactory):
    """Vehicle capacity must not be exceeded."""
    return (factory.for_each(Vehicle)
            .filter(lambda v: v.calculate_total_demand() > v.capacity)
            .penalize(HardSoftScore.ONE_HARD,
                      lambda v: v.calculate_total_demand() - v.capacity)
            .as_constraint("vehicleCapacity"))

def service_time_windows(factory: ConstraintFactory):
    """Service must finish within time windows."""
    return (factory.for_each(Visit)
            .filter(lambda visit: visit.is_service_finished_after_max_end_time())
            .penalize(HardSoftScore.ONE_HARD,
                      lambda visit: visit.service_finished_delay_in_minutes())
            .as_constraint("serviceTimeWindows"))

def minimize_unassigned_visits(factory: ConstraintFactory):
    """Penalize unassigned visits heavily."""
    return (factory.for_each(Visit)
            .filter(lambda visit: visit.vehicle is None)
            .penalize(HardSoftScore.of(0, 1000000))
            .as_constraint("minimizeUnassignedVisits"))

def minimize_total_travel_time(factory: ConstraintFactory):
    """Minimize total travel time across all vehicles."""
    return (factory.for_each(Vehicle)
            .filter(lambda v: len(v.visits) > 0)
            .penalize(HardSoftScore.ONE_SOFT,
                      lambda v: v.calculate_total_driving_time_seconds())
            .as_constraint("minimizeTotalTravelTime"))