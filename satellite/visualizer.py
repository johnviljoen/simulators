import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from datetime import datetime
import os
from common.rotation import quaternion_to_euler, euler_to_rot_matrix
import common.pytorch_utils as ptu

class Animator:
    def __init__(
            self,
            states,
            references,
            times,
            sphere_rad=1.0,
            save=True,
            save_path="data/media"
        ) -> None:
        
        self.states = states
        self.references = references
        self.times = times
        self.sphere_rad = sphere_rad
        self.save = save
        self.save_path = save_path

        # Unpack States for readability
        # -----------------------------

        self.x = states[:,0]
        self.y = states[:,1]
        self.z = -states[:,2]

        self.q0 = states[:,3]
        self.q1 = states[:,4]
        self.q2 = states[:,5]
        self.q3 = states[:,6]

        # Instantiate the figure with title, time, limits...
        # --------------------------------------------------

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        self.xDes = references[:, 0]
        self.yDes = references[:, 1]
        self.zDes = references[:, 2]
        self.ax.plot(self.xDes, self.yDes, self.zDes, ':', lw=1.3, color='green')

        self.sphere = None # the sphere is only drawn later
        self.line1, = self.ax.plot([], [], [], lw=2, color='black') # the line that will be the orbit
        self.line2, = self.ax.plot([], [], [], lw=2, color='blue') # the line that will be the orbit
        self.line3, = self.ax.plot([], [], [], lw=2, color='green') # the line that will be the orbit
        self.line4, = self.ax.plot([], [], [], lw=2, color='red') # the line that will be the orbit
        
        # Setting the axes properties
        extraEachSide = 0.5

        # Setting the axes properties
        extraEachSide = 0.5

        x_min = min(np.min(self.x), np.min(self.xDes))
        y_min = min(np.min(self.y), np.min(self.yDes))
        z_min = min(np.min(self.z), np.min(self.zDes))
        x_max = max(np.max(self.x), np.max(self.xDes))
        y_max = max(np.max(self.y), np.max(self.yDes))
        z_max = max(np.max(self.z), np.max(self.zDes))

        maxRange = 0.5*np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() + extraEachSide
        mid_x = 0.5*(x_max+x_min)
        mid_y = 0.5*(y_max+y_min)
        mid_z = 0.5*(z_max+z_min)

        self.length = maxRange * 0.1
        
        self.ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange]) # NED inv
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
        self.ax.set_zlabel('Z')

        self.titleTime = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)
        title = self.ax.text2D(0.95, 0.95, 'test_title', transform=self.ax.transAxes, horizontalalignment='right')


    def draw_sphere(self, x_center=0, y_center=0, z_center=0, radius=1, resolution=100):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x_grid = radius * np.outer(np.cos(u), np.sin(v)) + x_center
        y_grid = radius * np.outer(np.sin(u), np.sin(v)) + y_center
        z_grid = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center
        return self.ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=5, cstride=5, color='b')        
        
    def update_lines(self, i):

        # we draw this every self.num_frames
        time = self.times[i]

        x = self.x[i]
        y = self.y[i]
        z = self.z[i]

        # to draw the history of the line so far we need to retrieve all of it
        x_from0 = self.x[0:i]
        y_from0 = self.y[0:i]
        z_from0 = self.z[0:i]

        # retrieve quat
        quat = np.array([
            self.q0,
            self.q1,
            self.q2,
            self.q3
        ])[:,i]

        eul = quaternion_to_euler.numpy(quat)

        # remove the old cylinder if it exists
        if self.sphere is not None:
            self.sphere.remove()
            self.sphere = None

        # Draw a cylinder at a given location
        if self.sphere is None:
            self.sphere = self.draw_sphere(x_center=0, y_center=0, z_center=0, radius=self.sphere_rad)

        # the line drawing the trajectory itself
        self.line1.set_data(x_from0, y_from0)
        self.line1.set_3d_properties(z_from0)

        # remove the old quivers if they exist
        self.quiver_x.remove()
        self.quiver_y.remove()
        self.quiver_z.remove()

        # Draw new quivers at the updated location
        roll, pitch, yaw = eul
        R = euler_to_rot_matrix.numpy(roll, pitch, yaw)
        length = self.length  # adjust as necessary
        x_axis = R @ np.array([length, 0, 0])
        y_axis = R @ np.array([0, length, 0])
        z_axis = R @ np.array([0, 0, length])

        self.quiver_x = self.ax.quiver(x, y, z, x_axis[0], x_axis[1], x_axis[2], color='r', length=length, normalize=True)
        self.quiver_y = self.ax.quiver(x, y, z, y_axis[0], y_axis[1], y_axis[2], color='g', length=length, normalize=True)
        self.quiver_z = self.ax.quiver(x, y, z, z_axis[0], z_axis[1], z_axis[2], color='b', length=length, normalize=True)
        
        # the line drawing the 
        self.titleTime.set_text(u"Time = {:.2f} s".format(time))

        return self.line1, self.quiver_x, self.quiver_y, self.quiver_z
    
    def ini_plot(self):

        # trajectory line instantiation
        self.line1.set_data(np.empty([1]), np.empty([1]))
        self.line1.set_3d_properties(np.empty([1]))

        # Instantiate quivers for 3 orientation lines
        self.quiver_x = self.ax.quiver(0, 0, 0, 0, 0, 0, color='r', length=1, normalize=True)
        self.quiver_y = self.ax.quiver(0, 0, 0, 0, 0, 0, color='g', length=1, normalize=True)
        self.quiver_z = self.ax.quiver(0, 0, 0, 0, 0, 0, color='b', length=1, normalize=True)

        return self.line1, self.quiver_x, self.quiver_y, self.quiver_z
    
    def animate(self):
        line_ani = animation.FuncAnimation(
            self.fig, 
            self.update_lines, 
            init_func=self.ini_plot, 
            # frames=len(self.times[0:-2:self.num_frames]), 
            frames=len(self.times)-1, 
            interval=((self.times[1] - self.times[0])*10), 
            blit=False)

        if self.save is True:
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"save path: {os.path.abspath(self.save_path)}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            line_ani.save(f'{self.save_path}/{current_datetime}.gif', dpi=120, fps=25)
            # Update the figure with the last frame of animation
            self.update_lines(len(self.times[1:])-1)
            # Save the final frame as an SVG for good paper plots
            self.fig.savefig(f'{self.save_path}/{current_datetime}_final_frame.svg', format='svg')

        # plt.close(self.fig)            
        # plt.show()
        return line_ani
    
if __name__ == "__main__":

    # EXAMPLE USAGE
    # -------------

    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters
    num_points = 100  # Number of points (100 seconds / 0.1s intervals)
    radius_state = 5  # Arbitrary radius for the state circle
    radius_ref = 7  # Arbitrary radius for the reference circle
    angle_increment = 2 * np.pi / num_points  # Incremental angle for full circle

    # Generate times
    times = np.linspace(0, 100, num_points)

    # Generate states and references
    states = np.zeros((num_points, 7))  # (x, y, z, q0, q1, q2, q3)
    references = np.zeros((num_points, 3))  # (x, y, z)

    for i, time in enumerate(times):
        # State circle in the XY plane
        states[i, 0] = radius_state * np.cos(angle_increment * i)  # x
        states[i, 1] = radius_state * np.sin(angle_increment * i)  # y
        states[i, 2] = -radius_state * np.sin(angle_increment * i * 0.5)  # z, some variation in z to make it 3D

        # Reference circle in the YZ plane
        references[i, 0] = 0  # x
        references[i, 1] = radius_ref * np.sin(angle_increment * i)  # y
        references[i, 2] = radius_ref * np.cos(angle_increment * i)  # z

        # Generate a rotating quaternion (simplified for demonstration)
        angle = angle_increment * i
        states[i, 3] = np.cos(angle / 2)  # q0
        states[i, 4] = 0  # q1, assume rotation about Z-axis only
        states[i, 5] = 0  # q2
        states[i, 6] = np.sin(angle / 2)  # q3

    animator = Animator(states, references, times)
    animator.animate()

    print('fin')