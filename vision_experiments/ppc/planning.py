import numpy as np

class FollowTheGapPlanner:
    """
    Process LiDAR data to replace infinity with 0 and find a longest range
    of gaps based on a threshold distance, then steer towards the middle of
    this gap. Normalise the calculated angle into the right range.
    """
    def __init__(self, threshold_distance=3.0):  # Increased threshold distance
        self.threshold_distance = threshold_distance

    def preprocess_lidar(self, ranges):
        processed_ranges = np.array(ranges)
        processed_ranges = np.where(np.isinf(processed_ranges), 0, processed_ranges)
        return processed_ranges

    def find_max_gap(self, free_space_ranges):
        gaps = []
        start = 0
        while start < len(free_space_ranges):
            if free_space_ranges[start] > self.threshold_distance:
                end = start
                while end < len(free_space_ranges) and free_space_ranges[end] > self.threshold_distance:
                    end += 1
                gaps.append((start, end - 1))
                start = end
            else:
                start += 1
        max_gap = max(gaps, key=lambda gap: gap[1] - gap[0])
        return max_gap

    def find_best_point(self, start_idx, end_idx):
        return (start_idx + end_idx) // 2

    def plan_path(self, perception_data):
        lidar_ranges = perception_data['lidar_ranges']
        angle_increment = 2 * np.pi / len(lidar_ranges)

        # Preprocess the LIDAR scan data to fill any gaps
        processed_ranges = self.preprocess_lidar(lidar_ranges)

        # Find the maximum gap in the LIDAR data
        max_gap = self.find_max_gap(processed_ranges)

        # Find the best point to navigate to within the maximum gap
        best_point_index = self.find_best_point(max_gap[0], max_gap[1])
        best_point_angle = best_point_index * angle_increment - np.pi  # Adjust to -pi to pi range

        # Normalize the angle to [-1, 1]
        normalized_angle = best_point_angle / np.pi

        return normalized_angle, processed_ranges

def plan_path(perception_data):
    planner = FollowTheGapPlanner()
    steering_angle, processed_ranges = planner.plan_path(perception_data)
    return steering_angle, processed_ranges