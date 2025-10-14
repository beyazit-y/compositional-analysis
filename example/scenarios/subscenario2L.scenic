import numpy as np

param use2DMap = True
param map = "/Users/beyazit/Documents/compositional-analysis/Scenic/assets/maps/CARLA/Town07.xodr"

model scenic.simulators.metadrive.model

TARGET_SPEED = VerifaiRange(3, 12)
ego_speed = VerifaiRange(3, 12)
DISTANCE_TO_INTERSECTION = VerifaiRange(-15, -5)

# Ego vehicle just follows the trajectory specified later on.
behavior EgoBehavior(trajectory):
    do FollowTrajectoryBehavior(trajectory=trajectory, target_speed=TARGET_SPEED)
    terminate

# Find all 4-way intersections and set up trajectories for each vehicle.
fourWayIntersection = filter(lambda i: i.is4Way, network.intersections)

# choose intersection
intersec = fourWayIntersection[0] # choose one
# intersec = Uniform(*fourWayIntersection) # random

rightLanes = filter(lambda lane: all([section._laneToRight is None for section in lane.sections]), intersec.incomingLanes)
startLane = rightLanes[0] # choose one
# startLane = Uniform(*rightLanes) # random

left_maneuvers = filter(lambda i: i.type == ManeuverType.LEFT_TURN, startLane.maneuvers)
left_maneuver = Uniform(*left_maneuvers)

# go straight until intersection, stop at intersection
ego_trajectory = [left_maneuver.connectingLane, left_maneuver.endLane]

# Spawn each vehicle in the middle of its starting lane.
uberSpawnPoint = startLane.centerline[-1]

yaw, pitch, roll = roadDirection.value(uberSpawnPoint).eulerAngles

ego_vx = ego_speed * np.sin(yaw) * np.cos(pitch)
ego_vy = ego_speed * np.cos(yaw) * np.cos(pitch)
ego_vz = -ego_speed * np.sin(pitch)

ego = new Car following roadDirection from uberSpawnPoint for DISTANCE_TO_INTERSECTION,
        with behavior EgoBehavior(trajectory = ego_trajectory), with velocity ego_vx @ ego_vy

record ego.velocity.x as ego_vx
record ego.velocity.y as ego_vy
record ego.position as ego_position

