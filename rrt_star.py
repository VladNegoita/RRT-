from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random
import argparse


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0


class RRTStarHelpers:
    def __init__(
        self,
        start,
        goal,
        num_obstacles,
        map_size,
        step_size=0.2,
        max_iter=3000,
    ):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.map_size = map_size
        self.obstacles = self.generate_random_obstacles(num_obstacles)
        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.goal_region_radius = 0.4
        self.search_radius = 2.0

        self.path: None | List = None
        self.goal_reached = False
        self.best_node = Node(0, 0)

        # For MI-RRT*
        self.u = math.hypot(start[0] - goal[0], start[1] - goal[1])
        self.nu, self.c_prev, self.p_k = 0.999, 10 ** 7, 1.0

        self.fig, self.ax = plt.subplots(figsize=(8, 8), dpi=100)
        self.setup_visualization()

    def generate_random_obstacles(self, num_obstacles):
        """Randomly generate obstacles with random positions and sizes."""
        obstacles = []
        for _ in range(num_obstacles):
            ox = random.uniform(2, self.map_size[0] - 2)
            oy = random.uniform(2, self.map_size[1] - 2)
            size = random.uniform(1, 3.85)
            obstacles.append((ox, oy, size))
        return obstacles

    def setup_visualization(self):
        """Set up the visualization environment (grid, start, goal, obstacles)."""
        self.ax.plot(self.start.x, self.start.y, "bo", label="Start")
        self.ax.plot(self.goal.x, self.goal.y, "go", label="Goal")
        self.ax.set_xlim(0, self.map_size[0])
        self.ax.set_ylim(0, self.map_size[1])
        self.ax.grid(True)
        self.ax.legend(loc="lower right")

        # Draw the randomly generated obstacles
        self.draw_obstacles()

    def draw_obstacles(self):
        """Draw the static obstacles on the map."""
        for ox, oy, size in self.obstacles:
            circle = plt.Circle((ox, oy), size, color="r")
            self.ax.add_artist(circle)

    def get_random_node(self):
        """Generate a random node in the map."""
        while True:
            x = random.uniform(0, self.map_size[0])
            y = random.uniform(0, self.map_size[1])
            node = Node(x, y)
            if self.is_collision_free(node):
                return node

    def sample_unit_ball(self):
        r = math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)
        return (r * math.cos(theta), r * math.sin(theta))

    def transform_to_world(self, x_ball, r1, r2):
        x_scaled = (r1 * x_ball[0], r2 * x_ball[1])
        dx = self.goal.x - self.start.x
        dy = self.goal.y - self.start.y
        theta = math.atan2(dy, dx)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_rot = cos_t * x_scaled[0] - sin_t * x_scaled[1]
        y_rot = sin_t * x_scaled[0] + cos_t * x_scaled[1]
        return (x_rot, y_rot)

    def get_elipsoid_params(self):
        c_min = (
            math.sqrt(
                (self.start.x - self.goal.x) ** 2 + (self.start.y - self.goal.y) ** 2
            )
            - self.goal_region_radius
        )
        c_best = self.best_node.cost
        r1 = c_best / 2
        r2 = math.sqrt(c_best**2 - c_min**2) / 2
        return r1, r2

    def get_informed_random_node(self):
        r1, r2 = self.get_elipsoid_params()
        while True:
            x_ball = self.sample_unit_ball()
            dx, dy = self.transform_to_world(x_ball, r1, r2)
            new_x = (self.start.x + self.goal.x) / 2 + dx
            new_y = (self.start.y + self.goal.y) / 2 + dy
            if (
                new_x < self.map_size[0]
                and new_x > 0
                and new_y < self.map_size[1]
                and new_y > 0
            ):
                return Node(new_x, new_y)

    def get_mixed_random_node(self):
        if not self.path:
            raise RuntimeError("Trying to local sample without solution!")

        while True:
            px, py = random.choice(self.path)
            dx, dy = self.sample_unit_ball()
            c_k = self.best_node.cost
            R = self.step_size * min(c_k - self.u, 1.0)
            new_x = px + dx * R
            new_y = py + dy * R
            node = Node(new_x, new_y)
            if (
                new_x > 0
                and new_y > 0
                and new_x < self.map_size[0]
                and new_y < self.map_size[1]
                and self.is_collision_free(node)
            ):
                return node

    def steer(self, from_node, to_node):
        """Steer from one node to another, step-by-step."""
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(
            from_node.x + self.step_size * math.cos(theta),
            from_node.y + self.step_size * math.sin(theta),
        )
        new_node.cost = from_node.cost + self.step_size
        new_node.parent = from_node
        return new_node

    def is_collision_free(self, node):
        """Check if the node is collision-free with respect to obstacles."""
        for ox, oy, size in self.obstacles:
            if (node.x - ox) ** 2 + (node.y - oy) ** 2 <= size**2:
                return False
        return True

    def find_neighbors(self, new_node):
        """Find nearby nodes within the search radius."""
        return [
            node
            for node in self.node_list
            if np.linalg.norm([node.x - new_node.x, node.y - new_node.y])
            < self.search_radius
        ]

    def choose_parent(self, neighbors, nearest_node, new_node):
        """Choose the best parent for the new node based on cost."""
        min_cost = nearest_node.cost + np.linalg.norm(
            [new_node.x - nearest_node.x, new_node.y - nearest_node.y]
        )
        best_node = nearest_node

        for neighbor in neighbors:
            cost = neighbor.cost + np.linalg.norm(
                [new_node.x - neighbor.x, new_node.y - neighbor.y]
            )
            if cost < min_cost and self.is_collision_free(neighbor):
                best_node = neighbor
                min_cost = cost

        new_node.cost = min_cost
        new_node.parent = best_node
        return new_node

    def rewire(self, new_node, neighbors):
        """Rewire the tree by checking if any neighbor should adopt the new node as a parent."""
        for neighbor in neighbors:
            cost = new_node.cost + np.linalg.norm(
                [neighbor.x - new_node.x, neighbor.y - new_node.y]
            )
            if cost < neighbor.cost and self.is_collision_free(neighbor):
                neighbor.parent = new_node
                neighbor.cost = cost

    def reached_goal(self, node):
        """Check if the goal has been reached."""
        return (
            np.linalg.norm([node.x - self.goal.x, node.y - self.goal.y])
            < self.goal_region_radius
        )

    def get_best_path(self):
        goal = min(
            [node for node in self.node_list if self.reached_goal(node)],
            key=lambda x: x.cost,
        )
        return self.generate_final_path(goal)

    def generate_final_path(self, goal_node):
        """Generate the final path from the start to the goal."""
        path = []
        node = goal_node
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]  # Reverse the path

    def get_nearest_node(self, node_list, rand_node):
        """Find the nearest node in the tree to the random node."""
        distances = [
            np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y])
            for node in node_list
        ]
        nearest_node = node_list[np.argmin(distances)]
        return nearest_node

    def draw_tree(self, node):
        """Draw a tree edge from the current node to its parent."""
        if node.parent:
            self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b")

    def draw_path(self, how="-g"):
        """Draw the final path from start to goal."""
        if self.path:
            self.ax.plot(
                [x[0] for x in self.path], [x[1] for x in self.path], how, label="Path"
            )


def main():
    random.seed(69)

    start = [1, 1]
    goal = [15, 15]
    num_obstacles = 10
    map_size = [30, 30]

    rrt_star_helpers = RRTStarHelpers(start, goal, num_obstacles, map_size)

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo")
    args = parser.parse_args()

    def update_rrt_star(i: int):
        rrt_star_helpers.ax.set_title(
            f"RRT* - frame: {i}, cost: {rrt_star_helpers.best_node.cost if rrt_star_helpers.best_node.cost != 0 else 'inf'}"
        )
        if i < rrt_star_helpers.max_iter:
            rand_node = rrt_star_helpers.get_random_node()
            nearest_node = rrt_star_helpers.get_nearest_node(
                rrt_star_helpers.node_list, rand_node
            )
            new_node = rrt_star_helpers.steer(nearest_node, rand_node)

            if rrt_star_helpers.is_collision_free(new_node):
                neighbors = rrt_star_helpers.find_neighbors(new_node)
                new_node = rrt_star_helpers.choose_parent(
                    neighbors, nearest_node, new_node
                )
                rrt_star_helpers.node_list.append(new_node)
                rrt_star_helpers.rewire(new_node, neighbors)
                rrt_star_helpers.draw_tree(new_node)

            if rrt_star_helpers.reached_goal(new_node):
                if (
                    not rrt_star_helpers.goal_reached
                    or rrt_star_helpers.best_node.cost > new_node.cost
                ):
                    rrt_star_helpers.best_node = new_node
                rrt_star_helpers.goal_reached = True
                rrt_star_helpers.path = rrt_star_helpers.generate_final_path(new_node)
                rrt_star_helpers.draw_path()

        if rrt_star_helpers.goal_reached and i >= rrt_star_helpers.max_iter - 1:
            rrt_star_helpers.path = rrt_star_helpers.get_best_path()
            rrt_star_helpers.path = rrt_star_helpers.draw_path("-y")

        return []

    def update_informed_rrt_star(i: int):
        rrt_star_helpers.ax.set_title(
            f"Informed-RRT* - frame: {i}, cost: {rrt_star_helpers.best_node.cost if rrt_star_helpers.best_node.cost != 0 else 'inf'}"
        )
        if i < rrt_star_helpers.max_iter:
            if rrt_star_helpers.goal_reached:
                rand_node = rrt_star_helpers.get_informed_random_node()
            else:
                rand_node = rrt_star_helpers.get_random_node()

            nearest_node = rrt_star_helpers.get_nearest_node(
                rrt_star_helpers.node_list, rand_node
            )
            new_node = rrt_star_helpers.steer(nearest_node, rand_node)

            if rrt_star_helpers.is_collision_free(new_node):
                neighbors = rrt_star_helpers.find_neighbors(new_node)
                new_node = rrt_star_helpers.choose_parent(
                    neighbors, nearest_node, new_node
                )
                rrt_star_helpers.node_list.append(new_node)
                rrt_star_helpers.rewire(new_node, neighbors)
                rrt_star_helpers.draw_tree(new_node)

            if rrt_star_helpers.reached_goal(new_node):
                if (
                    not rrt_star_helpers.goal_reached
                    or rrt_star_helpers.best_node.cost > new_node.cost
                ):
                    rrt_star_helpers.best_node = new_node

                rrt_star_helpers.goal_reached = True
                rrt_star_helpers.path = rrt_star_helpers.generate_final_path(new_node)
                rrt_star_helpers.draw_path()

        if rrt_star_helpers.goal_reached and i >= rrt_star_helpers.max_iter - 1:
            rrt_star_helpers.path = rrt_star_helpers.get_best_path()
            rrt_star_helpers.path = rrt_star_helpers.draw_path("-y")

        return []

    def update_mixed_rrt_star(i: int):
        rrt_star_helpers.ax.set_title(
            f"MI-RRT* - frame: {i}, cost: {rrt_star_helpers.best_node.cost if rrt_star_helpers.best_node.cost != 0 else 'inf'}"
        )
        if i < rrt_star_helpers.max_iter:
            if not rrt_star_helpers.goal_reached:
                rand_node = rrt_star_helpers.get_random_node()
            elif random.random() < rrt_star_helpers.p_k:
                rand_node = rrt_star_helpers.get_mixed_random_node()
            else:
                rand_node = rrt_star_helpers.get_informed_random_node()

            nearest_node = rrt_star_helpers.get_nearest_node(
                rrt_star_helpers.node_list, rand_node
            )
            new_node = rrt_star_helpers.steer(nearest_node, rand_node)

            if rrt_star_helpers.is_collision_free(new_node):
                neighbors = rrt_star_helpers.find_neighbors(new_node)
                new_node = rrt_star_helpers.choose_parent(
                    neighbors, nearest_node, new_node
                )
                rrt_star_helpers.node_list.append(new_node)
                rrt_star_helpers.rewire(new_node, neighbors)
                rrt_star_helpers.draw_tree(new_node)
 
            if rrt_star_helpers.reached_goal(new_node):
                if (
                    not rrt_star_helpers.goal_reached
                    or rrt_star_helpers.best_node.cost > new_node.cost
                ):
                    rrt_star_helpers.best_node = new_node

                if not rrt_star_helpers.goal_reached:
                    print('Goal reached!')

                rrt_star_helpers.goal_reached = True
                rrt_star_helpers.path = rrt_star_helpers.generate_final_path(new_node)
                rrt_star_helpers.draw_path()

                c_k = rrt_star_helpers.best_node.cost
                if c_k < rrt_star_helpers.c_prev:
                    delta = (rrt_star_helpers.c_prev - c_k) / (
                        rrt_star_helpers.c_prev - rrt_star_helpers.u
                    )
                    rrt_star_helpers.p_k = (
                        rrt_star_helpers.nu * rrt_star_helpers.p_k
                        + (1 - rrt_star_helpers.nu) * delta
                    )
                    rrt_star_helpers.c_prev = c_k
                else:
                    rrt_star_helpers.p_k = rrt_star_helpers.nu * rrt_star_helpers.p_k

        if rrt_star_helpers.goal_reached and i >= rrt_star_helpers.max_iter - 1:
            rrt_star_helpers.path = rrt_star_helpers.get_best_path()
            rrt_star_helpers.path = rrt_star_helpers.draw_path("-y")

        return []

    filename, update_func = "rrt_star.mp4", update_rrt_star
    if args.algo == "informed":
        filename, update_func = "informed_rrt_star.mp4", update_informed_rrt_star
    elif args.algo == "mixed":
        filename, update_func = "mixed_rrt_star.mp4", update_mixed_rrt_star

    ani = animation.FuncAnimation(
        rrt_star_helpers.fig,
        update_func,
        frames=rrt_star_helpers.max_iter + 30,
        interval=0.000000001,
        repeat=False,
    )

    ani.save(filename=filename, writer=animation.FFMpegWriter(fps=240))
    print(f"Finished saving animation as {filename}")


if __name__ == "__main__":
    main()
