import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RRTNode:
    """Node for RRT* tree"""
    def __init__(self, position, parent=None):
        self.position = np.array(position, dtype=float)
        self.parent = parent
        self.cost = 0.0
        self.children = []

class PathPlanner:
    """
    Robust RRT* implementation for 3D path planning
    """
    
    def __init__(self, environment):
        self.env = environment
        self.waypoints = []
        self.tree_nodes = []
        
        # RRT* parameters
        self.max_iterations = 3000
        self.step_size = 1.0
        self.goal_radius = 1.0
        self.search_radius = 2.5
        self.goal_bias = 0.15  # 15% bias towards goal
        
    
    ############################################################################################################
    #### TODO - Implement RRT* path planning algorithm in 3D (use the provided environment class) ##############
    #### TODO - Store the final path in self.waypoints as a list of 3D points ##################################
    #### TODO - Add member functions as needed #################################################################
    ############################################################################################################


    
    def visualize_tree(self, ax=None):
        """Visualize the RRT* tree"""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            standalone = True
        else:
            standalone = False
        
        # Draw tree edges
        for node in self.tree_nodes:
            if node.parent is not None:
                ax.plot([node.parent.position[0], node.position[0]],
                       [node.parent.position[1], node.position[1]],
                       [node.parent.position[2], node.position[2]],
                       'b-', alpha=0.3, linewidth=0.5)
        
        # Draw tree nodes
        if self.tree_nodes:
            positions = np.array([node.position for node in self.tree_nodes])
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='blue', s=10, alpha=0.6)
        
        # Draw final path
        if len(self.waypoints) > 0:
            waypoints = np.array(self.waypoints)
            ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                   'ro-', markersize=8, linewidth=3, label='RRT* Path')
        
        if standalone:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('RRT* Tree and Path')
            ax.legend()
            plt.tight_layout()
            plt.show()
        
        return ax
    

    # Used from environment.py 
    # step 1 def generate_random_free_point(self):
    # step 2 def is_point_in_free_space(self, point):
    # step 3 find nearest_node in tree to new generated node
    # step 4 distance b/w nearest_node and  new_node
    # step 5 find_neighbours within search radius for new_node and existing nodes of tree
    # step 6 connect x_best and X_new (make an edge)
    # step 7 update X_new in tree_nodes

    # helper function 
    # def is_line_collision_free(self, p1, p2, num_checks=20):

    # def nearest_node(self, new_node, tree_nodes):
        
    #     # return nearest node in tree to new node 
    
    # def distance(self, nearest_node, new_node):

    #     #return distance b/w nearest_node and  new_node
    
    # def find_neighbours(self, new_node, tree_nodes):

    #     #return x_best, x_neighbours
    # def update_tree(self, new_node,x_neighbours, tree_nodes):

    #     #return add x_new to to lowest cost member of x_neighbours and return new tree
    

    def find_nearest_node(self, tree_nodes, sample_point):
        """
        Find nearest node in the tree to the sample point
        return nearest_node 
        """
        if not tree_nodes:
            return None
        
        positions = np.array([node.position for node in tree_nodes])
        distances=np.linalg.norm(positions - sample_point,axis=1)
        nearest_index=np.argmin(distances)   
        nearest_node=tree_nodes[nearest_index]


        return nearest_node
            

    def steer(self, from_point, sample_point, step_size):
        """
        Find new  point step_size away in direction of sample_point from nearest_node
        return new point
        """
        #making 2 vectors 
        x_1=np.array(from_point)
        x_2=np.array(sample_point)

        #direction b/w vectors
        direction_vec=x_2 -x_1

        total_length= np.linalg.norm(direction_vec)

        if total_length == 0:
            return x_1
        if total_length < step_size:
            return sample_point
        
        unit_vec= direction_vec/total_length

        new_point= x_1 + unit_vec*step_size

        return new_point


    def find_near_nodes(self, tree, new_position, search_radius):
        """
        Find all nodes in the tree within the search radius of the new position
        """

        near_node=[]
        for node in tree:

            #check only if within search radius
            if np.linalg.norm(new_position - node.position)< search_radius:
                near_node.append(node)
        return near_node
    

    
    def choose_parent(self, near_nodes, new_position):
        """
        Choose the best parent for the new node from the list of near nodes
        """

        min_cost=float('inf')
        best_parent=None

        for node in near_nodes:
            potential_cost=node.cost+ self.distance(node,new_position)

            if potential_cost<min_cost:
                if self.is_path_valid(node.position , new_position):
                    min_cost=potential_cost
                    best_parent=node

        return best_parent, min_cost



    def is_path_valid(self, p1, p2):
        """
        Check if the path between two points is collision-free
        """
        return self.env.is_line_collision_free(p1, p2)
    

    
    def distance(self, p1, p2):
        """
        Calculates the Euclidean distance between two points. 
        The points can be RRTNode objects or numpy arrays.
        """
       
        pos1 = p1.position if hasattr(p1, 'position') else p1
        pos2 = p2.position if hasattr(p2, 'position') else p2
        
        return np.linalg.norm(pos1 - pos2)
        
    

    def rewire_tree(self, tree, new_node, near_nodes):
        for node in near_nodes:
            
            # Dont rewire parent of new node 
            if node==new_node.parent:
                continue
            
            # 
            potential_cost=node.cost+ self.distance(node.position ,new_node.position)
            if potential_cost<node.cost:
                if self.is_path_valid(node.position , new_node.position):
                    if node.parent:
                        node.parent.children.remove(node)
                    node.parent = new_node
                    node.cost = potential_cost
                    new_node.children.append(node)
        
        return tree
        

    def extract_path(self, goal_node):
        """
        Extract the path from start to goal by backtracking from the goal node
        """

        path=[]
        node = goal_node
        while node is not None:
            path.append(node.position.tolist())
            node = node.parent
        
        return path[::-1]  # Reverse the path

    def plan_path(
    self,
    max_iterations=None,
    step_size=None,
    goal_radius=None,
    search_radius=None,
    goal_bias=None,
):
        """
        Build an RRT* path using existing helpers and store in self.waypoints.
        Returns True on success, False otherwise.
        Optional args override the planner defaults.
        """
        # Sanity checks
        if not self.env.boundary or self.env.start_point is None or self.env.goal_point is None:
            return False

        start = np.array(self.env.start_point, dtype=float)
        goal  = np.array(self.env.goal_point,  dtype=float)
        if not self.env.is_point_in_free_space(start) or not self.env.is_point_in_free_space(goal):
            return False

        # Parameters (adaptive defaults)
        if max_iterations is None:
            max_iterations = self.max_iterations

        xmin, ymin, zmin, xmax, ymax, zmax = self.env.boundary
        map_diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
        sg_dist  = np.linalg.norm(goal - start)

        if step_size is None:
            step_size = float(np.clip(sg_dist / 12.0, 0.15, 0.6))
        if goal_radius is None:
            goal_radius = float(np.clip(sg_dist / 15.0, 0.25, 0.7))
        if search_radius is None:
            search_radius = 2.5 * step_size
        if goal_bias is None:
            goal_bias = self.goal_bias

        # Grow RRT*
        self.tree_nodes = []
        start_node = RRTNode(start)
        self.tree_nodes.append(start_node)
        goal_node = None

        for _ in range(max_iterations):
            # Sample with goal bias
            if np.random.rand() < goal_bias:
                sample = goal
            else:
                p = self.env.generate_random_free_point()
                if p is None:
                    continue
                sample = np.array(p, dtype=float)

            # Nearest
            nearest = self.find_nearest_node(self.tree_nodes, sample)
            if nearest is None:
                continue

            # Steer
            new_pos = self.steer(nearest.position, sample, step_size)

            # Validity
            if not self.env.is_point_in_free_space(new_pos):
                continue
            if not self.env.is_line_collision_free(nearest.position, new_pos):
                continue

            # Near set & choose parent
            near = self.find_near_nodes(self.tree_nodes, new_pos, search_radius)
            best_parent, best_cost = self.choose_parent(near, new_pos)
            if best_parent is None:
                if not self.is_path_valid(nearest.position, new_pos):
                    continue
                best_parent = nearest
                best_cost = nearest.cost + self.distance(nearest.position, new_pos)

            # Add node
            node = RRTNode(new_pos, parent=best_parent)
            node.cost = best_cost
            best_parent.children.append(node)
            self.tree_nodes.append(node)

            # Rewire
            self.rewire_tree(self.tree_nodes, node, near)

            # Goal check
            if self.distance(new_pos, goal) <= goal_radius and self.is_path_valid(new_pos, goal):
                goal_node = RRTNode(goal, parent=node)
                goal_node.cost = node.cost + self.distance(new_pos, goal)
                node.children.append(goal_node)
                self.tree_nodes.append(goal_node)
                break

        if goal_node is None:
            return False

        # Extract (& optionally simplify)
        self.waypoints = self.extract_path(goal_node)
        self.waypoints = self.simplify_path(self.waypoints)
        return True



    

    def simplify_path(self, waypoints):
        """
        Simplify the path using a straightforward line-of-sight check.
        """
        if len(waypoints) < 3:
            return waypoints

        # Start the simplified path with the first waypoint
        simplified_path = [waypoints[0]]

        # Iterate through the waypoints to find points that can be skipped
        for i in range(1, len(waypoints) - 1):
            # Check if we can go directly from the last point in our simplified path
            # to the point *after* the current one.
            if not self.env.is_line_collision_free(simplified_path[-1], waypoints[i + 1]):
                # If we can't skip the current point, we must add it.
                simplified_path.append(waypoints[i])
        
        # Always add the very last waypoint (the goal)
        simplified_path.append(waypoints[-1])

        return simplified_path
    
    # testing for bspline 
    # def simplify_path(self,waypoints):
    #     return waypoints