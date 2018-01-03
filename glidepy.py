import numpy as np
from scipy.optimize import fmin
import math
from scipy.optimize import minimize


class Glider:

    def __init__(self, speeds, sink_rates, weight_ref, weight=None):
        self._init_constants()
        # self.wm = 0
        self.altitude = 0
        self.flight_state = "on_ground"

        self.weight_ref = weight_ref
        self.weight = weight

        self.original_speeds = speeds
        self.original_sink_rates = sink_rates

        self._calc_scale_factor()
        self._scale_speeds()
        self._calc_polar_model()
        self.calc_stf_model()

    def _init_constants(self):
        self.kmh_to_knots = 0.539957
        self.ms_to_knots = 1.94384
        self.knots_to_kmh = 1.852
        self.nm_to_feet = 6076.12
        self.nm_to_sm = 1.15078

    def _calc_scale_factor(self):
        if self.weight is None:
            self._scale_factor = 1.0
        else:
            self._scale_factor = math.sqrt(self.weight / self.weight_ref)

    def _scale_speeds(self):
        self.speeds = self.kmh_to_knots * self._scale_factor * np.array(self.original_speeds)
        self.sink_rates = self.ms_to_knots * self._scale_factor * np.array(self.original_sink_rates)

    def _calc_polar_model(self):
        self._polar_model = np.polyfit(self.speeds, self.sink_rates, 3)

    def _glide_ratio2(self, speed):
        return speed/self.sink_rate(speed)

    def best_ld(self):
        res = minimize(self._glide_ratio2, 0, method="SLSQP")
        return -res.fun

    def best_ld_speed(self):
        res = minimize(self._glide_ratio2, 0, method="SLSQP")
        return res.x[0]

    def calc_stf_model(self):
        distance = 1
        lower_limit = 0.0001
        climb_range = np.arange(lower_limit, 10.5, 1)
        stf_values = [fmin(self.calc_total_time, 1, args=(x, distance), disp=False)[0] for x in climb_range]
        self._stf_model = np.polyfit(climb_range, stf_values, 4)

    def sink_rate(self, speed):
        return self.polar(speed)

    def polar(self, speed):
        return np.polyval(self._polar_model, speed)

    def glide_ratio(self, speed):
        return -speed/self.sink_rate(speed)

    def calc_avg_speed(self, speed, climb_rate, e=1):
        total_time = self.calc_total_time(speed, climb_rate, e)
        avg_speed = e / total_time
        return avg_speed

    def calc_total_time(self, speed, climb_rate, e=1):
        ws = np.polyval(self._polar_model, speed)
        total_time = e * (-(ws + 0) / (speed * climb_rate) + (1 / speed))
        return total_time

    def speed_to_fly(self, climb_rate, explicit=False):
        if explicit:
            return fmin(self.calc_total_time, 1, args=[climb_rate], disp=False)[0]
        else:
            return np.polyval(self._stf_model, climb_rate)

    def altitude_lost(self, speed, distance):
        sink_rate = -(self.polar(speed))
        altitude_lost = sink_rate * distance/speed
        return altitude_lost, altitude_lost * self.nm_to_feet

    def get_range(self, altitude, speed):
        sink_rate = self.polar(speed)
        if sink_rate >= 0:
            range1 = float("inf")
        else:
            glide_time = altitude/self.nm_to_feet/(-sink_rate)
            range1 = speed * glide_time
        return range1

    def set_altitude(self, altitude):
        self.altitude = altitude
        self.flight_state = 'flying'

    def cruise(self, mc, distance):
        speed = self.speed_to_fly(mc)
        altitude_loss = self.altitude_lost(speed, distance)[1]
        if altitude_loss > self.altitude:
            self.altitude = 0
            self.flight_state = "on_ground"
        else:
            self.altitude = self.altitude - altitude_loss
        cruise_time = distance/speed
        return cruise_time

    def climb(self, climb_step_size, climb_rate):
        self.altitude += climb_step_size
        climb_time = climb_step_size/(climb_rate * self.nm_to_feet)
        return climb_time


class Thermals:
    def __init__(self, dt, distance_to_destination, distance_step):
        self.cum_dist = 0
        self.thermals = np.array([])
        while self.cum_dist <= distance_to_destination:
            d = np.random.exponential(dt, 1)[0]
            if d < distance_step:
                d += distance_step  # make it at least a min distance of distant_step
            # print(d)
            self.thermals = np.append(self.thermals, d)
            self.cum_dist += d
        self.cum_sum = np.cumsum(self.thermals)

    def is_thermal(self, dist, distance_step):
        for d in self.cum_sum:
            if (d > dist) & (d < (dist + distance_step)):
                return d
        return 0


class Test:
    def __init__(self):
        print("ok")


