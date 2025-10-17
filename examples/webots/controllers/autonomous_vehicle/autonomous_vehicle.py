import sys
from controller import Supervisor
from controller import Robot, Camera, Display, GPS, Lidar
from vehicle import Driver
import math
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1000):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        # self.TIME_STEP = 50
        self.TIME_STEP = int(self.getBasicTimeStep())

        self.camera = self.getDevice("camera")
        self.camera_width = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.camera_fov = self.camera.getFov()

        self.sick = self.getDevice("Sick LMS 291")
        self.sick_width = self.sick.getHorizontalResolution()
        self.sick_range = self.sick.getMaxRange()
        self.sick_fov = self.sick.getFov()

        self.display = self.getDevice("display")
        self.speedometer_image = self.display.imageLoad("speedometer.png")

        self.gps = self.getDevice("gps")

        self.motor = self.getDevice("steering_wheel_motor")

        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0, 255, shape=(64, 128, 4), dtype=np.uint8)

    def reset(self, seed=None):
        self.t = 0

        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()

        self.camera.enable(self.TIME_STEP)
        self.sick.enable(self.TIME_STEP)
        self.gps.enable(self.TIME_STEP)

        super().step(self.TIME_STEP)

        obs = self.get_obs()

        # Internals
        # super().step(self.TIME_STEP)

        # Open AI Gym generic
        return obs, {}

    def step(self, action):
        self.t += 1
        self.motor.setAcceleration(10.0)
        super().step(self.TIME_STEP)
        done = self.t >= self.max_episode_steps
        return self.get_obs(), 0, done, False, {}
        # # Execute the action
        # for wheel in self.__wheels:
        #     wheel.setVelocity(1.3 if action == 1 else -1.3)
        # super().step(self.__timestep)

        # # Observation
        # robot = self.getSelf()
        # endpoint = self.getFromDef("POLE_ENDPOINT")
        # self.state = np.array([robot.getPosition()[0], robot.getVelocity()[0],
        #                        self.__pendulum_sensor.getValue(), endpoint.getVelocity()[4]])

        # # Done
        # done = bool(
        #     self.state[0] < -self.x_threshold or
        #     self.state[0] > self.x_threshold or
        #     self.state[2] < -self.theta_threshold_radians or
        #     self.state[2] > self.theta_threshold_radians
        # )

        # # Reward
        # reward = 0 if done else 1

        # return self.state.astype(np.float32), reward, done, False, {}

    def get_obs(self):
        image = self.camera.getImage()
        return np.asarray(list(image), dtype=np.uint8).reshape(64, 128, -1)


def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    # Train
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)
    model.learn(total_timesteps=1e5)

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()



# """
# Autonomous vehicle controller example (Python version)
# Equivalent to Cyberbotics' C controller.
# """

# from controller import Robot, Camera, Display, GPS, Lidar
# from vehicle import Driver
# import math

# # --- Constants ---
# TIME_STEP = 50
# UNKNOWN = 99999.99

# # PID constants
# KP = 0.25
# KI = 0.006
# KD = 2

# FILTER_SIZE = 3

# # --- Global state ---
# PID_need_reset = False
# enable_collision_avoidance = False
# enable_display = False
# has_gps = False
# has_camera = False

# camera = None
# camera_width = -1
# camera_height = -1
# camera_fov = -1.0

# sick = None
# sick_width = -1
# sick_range = -1.0
# sick_fov = -1.0

# display = None
# speedometer_image = None

# gps = None
# gps_coords = [0.0, 0.0, 0.0]
# gps_speed = 0.0

# speed = 0.0
# steering_angle = 0.0
# manual_steering = 0
# autodrive = True


# def set_speed(kmh):
#     global speed
#     kmh = min(kmh, 250.0)
#     speed = kmh
#     print(f"Setting speed to {kmh} km/h")
#     driver.setCruisingSpeed(kmh)


# def set_steering_angle(wheel_angle):
#     global steering_angle
#     # rate limiting
#     delta = wheel_angle - steering_angle
#     if delta > 0.1:
#         wheel_angle = steering_angle + 0.1
#     elif delta < -0.1:
#         wheel_angle = steering_angle - 0.1

#     steering_angle = max(min(wheel_angle, 0.5), -0.5)
#     driver.setSteeringAngle(steering_angle)


# def color_diff(a, b):
#     return sum(abs(int(a[i]) - int(b[i])) for i in range(3))


# def process_camera_image(image):
#     image_np = np.asarray(list(image)).reshape(camera_height, camera_width, -1)
#     print(image_np)
#     print(image_np.shape)
#     # input()
#     """Return approximate angle of yellow line or UNKNOWN."""
#     num_pixels = camera_width * camera_height
#     REF = (95, 187, 203)  # BGR
#     sumx, pixel_count = 0, 0

#     for y in range(camera_height):
#         for x in range(camera_width):
#             idx = (y * camera_width + x) * 4
#             pixel = image[idx:idx+3]  # B, G, R
#             if color_diff(pixel, REF) < 30:
#                 sumx += x
#                 pixel_count += 1

#     if pixel_count == 0:
#         return UNKNOWN

#     return ((sumx / pixel_count) / camera_width - 0.5) * camera_fov


# def filter_angle(new_value):
#     if not hasattr(filter_angle, "values"):
#         filter_angle.values = [0.0] * FILTER_SIZE
#         filter_angle.first_call = True

#     if filter_angle.first_call or new_value == UNKNOWN:
#         filter_angle.values = [0.0] * FILTER_SIZE
#         filter_angle.first_call = False
#     else:
#         filter_angle.values[:-1] = filter_angle.values[1:]

#     if new_value == UNKNOWN:
#         return UNKNOWN
#     else:
#         filter_angle.values[-1] = new_value
#         return sum(filter_angle.values) / FILTER_SIZE


# def process_sick_data(sick_data):
#     """Return obstacle angle and distance."""
#     HALF_AREA = 20
#     sumx, collision_count, obstacle_dist = 0, 0, 0.0

#     for x in range(sick_width // 2 - HALF_AREA, sick_width // 2 + HALF_AREA):
#         r = sick_data[x]
#         if r < 20.0:
#             sumx += x
#             collision_count += 1
#             obstacle_dist += r

#     if collision_count == 0:
#         return UNKNOWN, 0.0

#     obstacle_dist /= collision_count
#     angle = ((sumx / collision_count) / sick_width - 0.5) * sick_fov
#     return angle, obstacle_dist


# def compute_gps_speed():
#     global gps_coords, gps_speed
#     coords = gps.getValues()
#     gps_coords = list(coords)
#     gps_speed = gps.getSpeed() * 3.6  # m/s → km/h


# def update_display():
#     global display, speedometer_image, gps_coords, gps_speed
#     NEEDLE_LENGTH = 50.0

#     # Display background
#     display.imagePaste(speedometer_image, 0, 0, False)

#     # Draw speedometer needle
#     current_speed = display.getDriver().getCurrentSpeed() if hasattr(display, "getDriver") else gps_speed
#     if math.isnan(current_speed):
#         current_speed = 0.0

#     alpha = current_speed / 260.0 * 3.72 - 0.27
#     x = -NEEDLE_LENGTH * math.cos(alpha)
#     y = -NEEDLE_LENGTH * math.sin(alpha)
#     display.drawLine(100, 95, int(100 + x), int(95 + y))

#     # Draw text
#     display.drawText(f"GPS coords: {gps_coords[0]:.1f} {gps_coords[2]:.1f}", 10, 130)
#     display.drawText(f"GPS speed:  {gps_speed:.1f}", 10, 140)


# def applyPID(yellow_line_angle):
#     global PID_need_reset
#     if not hasattr(applyPID, "oldValue"):
#         applyPID.oldValue = 0.0
#         applyPID.integral = 0.0

#     if PID_need_reset:
#         applyPID.oldValue = yellow_line_angle
#         applyPID.integral = 0.0
#         PID_need_reset = False

#     if math.copysign(1, yellow_line_angle) != math.copysign(1, applyPID.oldValue):
#         applyPID.integral = 0.0

#     diff = yellow_line_angle - applyPID.oldValue
#     if -30 < applyPID.integral < 30:
#         applyPID.integral += yellow_line_angle

#     applyPID.oldValue = yellow_line_angle
#     return KP * yellow_line_angle + KI * applyPID.integral + KD * diff


# # --- Initialization ---
# driver = Driver()
# robot = driver  # alias for consistency

# # Detect devices
# for i in range(robot.getNumberOfDevices()):
#     device = robot.getDeviceByIndex(i)
#     name = device.getName()
#     if name == "Sick LMS 291":
#         enable_collision_avoidance = True
#     elif name == "display":
#         enable_display = True
#     elif name == "gps":
#         has_gps = True
#     elif name == "camera":
#         has_camera = True

# # Camera
# if has_camera:
#     camera = robot.getDevice("camera")
#     camera.enable(TIME_STEP)
#     camera_width = camera.getWidth()
#     camera_height = camera.getHeight()
#     camera_fov = camera.getFov()

# # Lidar
# if enable_collision_avoidance:
#     sick = robot.getDevice("Sick LMS 291")
#     sick.enable(TIME_STEP)
#     sick_width = sick.getHorizontalResolution()
#     sick_range = sick.getMaxRange()
#     sick_fov = sick.getFov()

# # GPS
# if has_gps:
#     gps = robot.getDevice("gps")
#     gps.enable(TIME_STEP)

# # Display
# if enable_display:
#     display = robot.getDevice("display")
#     speedometer_image = display.imageLoad("speedometer.png")

# # Start engine
# if has_camera:
#     set_speed(50.0)
# driver.setHazardFlashers(True)
# driver.setDippedBeams(True)
# driver.setAntifogLights(True)
# driver.setWiperMode(Driver.SLOW)

# # --- Main loop ---
# i = 0
# while driver.step() != -1:

#     if i % int(TIME_STEP / robot.getBasicTimeStep()) == 0:
#         camera_image = camera.getImage() if has_camera else None
#         sick_data = sick.getRangeImage() if enable_collision_avoidance else None

#         if autodrive and has_camera:
#             yellow_line_angle = filter_angle(process_camera_image(camera_image))
#             if enable_collision_avoidance:
#                 obstacle_angle, obstacle_dist = process_sick_data(sick_data)
#             else:
#                 obstacle_angle, obstacle_dist = UNKNOWN, 0.0

#             if enable_collision_avoidance and obstacle_angle != UNKNOWN:
#                 driver.setBrakeIntensity(0.0)
#                 obstacle_steering = steering_angle
#                 if 0.0 < obstacle_angle < 0.4 and obstacle_dist > 1e-8:
#                     obstacle_steering += (obstacle_angle - 0.25) / obstacle_dist
#                 elif obstacle_angle > -0.4 and obstacle_dist > 1e-8:
#                     obstacle_steering += (obstacle_angle + 0.25) / obstacle_dist
#                 else:
#                     obstacle_steering += 0.0

#                 steer = steering_angle
#                 if yellow_line_angle != UNKNOWN:
#                     line_steer = applyPID(yellow_line_angle)
#                     if obstacle_steering > 0 and line_steer > 0:
#                         steer = max(obstacle_steering, line_steer)
#                     elif obstacle_steering < 0 and line_steer < 0:
#                         steer = min(obstacle_steering, line_steer)
#                 else:
#                     PID_need_reset = True
#                 set_steering_angle(steer)
#             elif yellow_line_angle != UNKNOWN:
#                 driver.setBrakeIntensity(0.0)
#                 set_steering_angle(applyPID(yellow_line_angle))
#             else:
#                 driver.setBrakeIntensity(0.4)
#                 PID_need_reset = True

#         if has_gps:
#             compute_gps_speed()
#         if enable_display:
#             update_display()
#         # Note: display update omitted for brevity

#     i += 1
