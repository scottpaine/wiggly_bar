from vispy import app, visuals
from vispy.visuals import transforms
import sys
import numpy as np


def make_spiral(num_points=100, num_turns=4, height=12, radius=2.0,
                xnot=None, ynot=None, znot=None):
    """
    Generate a list of points corresponding to a spiral.
    
    Parameters
    ----------
    num_points : int
        Number of points to map spiral over. More points means a rounder spring.
    num_turns : int
        Number of coils in the spiral
    height : float
        The height of the spiral. Keep it in whatever units the rest of the spiral is in.
    radius : float
        The radius of the coils. The spiral will end up being 2*radius wide.
    xnot : float
        Initial x-coordinate for the spiral coordinates to start at.
    ynot : float
        Initial y-coordinate for the spiral coordinates to start at.
    znot : float
        Initial z-coordinate for the spiral coordinates to start at.
    
    Returns
    -------
    coord_list: list of tuples
        Coordinate list of (x, y, z) positions for the spiral
        
    Notes
    -----
    Right now, this assumes the center is at x=0, y=0. Later, it might be good to add in stuff to change that.
    
    """

    coords_list = []
    znot = -4 if znot is None else znot
    xnot = radius if xnot is None else xnot
    ynot = 0 if ynot is None else ynot

    theta_not = np.arctan2(ynot, xnot)

    coords_list.append((xnot, ynot, znot))

    for point in range(num_points):
        znot += height / num_points
        theta_not += 2 * np.pi * num_turns / num_points
        xnot = np.cos(theta_not) * radius
        ynot = np.sin(theta_not) * radius
        coords_list.append((xnot, ynot, znot))
    return coords_list


def make_spring(num_points=300, num_turns=4, height=12, radius=2.0,
                xnot=None, ynot=None, znot=None):
    """
        Generate a list of points corresponding to a spring.
        
        Parameters
        ----------
        num_points : int
            Number of points to map spring over. More points means a rounder spring.
        num_turns : int
            Number of coils in the spring
        height : float
            The height of the spring. Keep it in whatever units the rest of the spring is in.
        radius : float
            The radius of the coils. The spring will end up being 2*radius wide.
        xnot : float
            Initial x-coordinate for the spring coordinates to start at.
        ynot : float
            Initial y-coordinate for the spring coordinates to start at.
        znot : float
            Initial z-coordinate for the spring coordinates to start at.
        
        Returns
        -------
        coord_list: list of tuples
            Coordinate list of (x, y, z) positions for the spring
            
        Notes
        -----
        Right now, this assumes the center is at x=0, y=0. Later, it might be good to add in stuff to change that.

        Right now, the length of the "ends" is 10% of the overall length, as well asa small "turn" that is length
        radius / 2. In the future, maybe there could be a kwarg to set the length of the sides of the spring. For
        now, 10% looks good.
        
        """
    coords_list = []
    init_pts = num_points // 10
    znot = 0 if znot is None else znot
    xnot = 0 if xnot is None else xnot
    ynot = 0 if ynot is None else ynot
    coords_list.append((xnot, ynot, znot))
    for _ in range(init_pts):
        znot += height / num_points
        coords_list.append((xnot, ynot, znot))
    hold_z = znot
    for i in range(init_pts // 2):
        small_theta = (i + 1) * np.pi / init_pts
        xnot = radius / 2 * (1 - np.cos(small_theta))
        znot = hold_z + radius / 2 * np.sin(small_theta)
        coords_list.append((xnot, ynot, znot))
    coords_list += make_spiral(num_points=num_points - 3 * init_pts,
                               num_turns=num_turns,
                               height=height - (91 * height / num_points) - radius / 2,
                               radius=radius,
                               xnot=xnot,
                               ynot=ynot,
                               znot=znot)
    hold_z = coords_list[-1][-1]
    for i in range(init_pts // 2):
        small_theta = np.pi / 2 - (i + 1) * np.pi / init_pts
        xnot = radius / 2 * (1 - np.cos(small_theta))
        znot = hold_z + radius / 2 * np.cos(small_theta)
        coords_list.append((xnot, ynot, znot))
    xnot = 0.0
    znot += height / num_points
    for _ in range(init_pts):
        znot += height / num_points
        coords_list.append((xnot, ynot, znot))
    coords_list.append((0, 0, height))

    return coords_list


class WigglyBar(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Wiggly Bar', size=(800, 800))

        # Initialize constants for the system
        self.t = 0
        self.d1 = 0.97
        self.d2 = 0.55
        self.little_m = 2.0
        self.big_m = 12.5
        self.spring_k1 = 1.35
        self.spring_k2 = 0.5
        self.b = 25.75
        self.j_term = (1/3)*self.big_m*(self.d1 ** 3 + self.d2 ** 3)/(self.d1 + self.d2)
        self.x = -0.010
        self.x_dot = -0.12
        self.theta = 0.005
        self.theta_dot = 0.0
        self.theta_not = self.theta  # I'll need this later

        # Initialize constants for display
        self.px_len = 10
        self.scale = 50
        self.center = np.asarray((500, 400 - 1.5*self.scale))
        self.ellipse_center = (500, 400)
        self.visuals = []
        self.px_per_m = self.scale * self.px_len/(self.d1 + self.d2)
        self.rod_length = self.scale * self.px_len/self.px_per_m

        # Set up stuff for establishing a pivot point to rotate about
        self.pivot_loc = (self.d1 - self.d2)/2
        self.pivot_loc_px = self.pivot_loc * self.px_per_m
        piv_x_y_px = np.asarray((
            -1*self.pivot_loc_px*np.sin(self.theta),
            self.pivot_loc_px*(1 - np.cos(self.theta))
        ))

        # Set up positioning info for the springs and mass, as well as some constants for use later
        # NOTE: Springs are not like boxes. Their center of rotation is at one end of the spring, unlike
        #       the box where it is in the middle. The location and scaling is set to reflect this. This means
        #       there's a little bit of x- and y-translation needed to properly center them.
        self.s2_loc = np.asarray(
            [self.d1 * self.px_per_m * np.sin(self.theta),
             self.px_len/8 * self.scale + self.px_len/100*self.scale - self.d1 * self.px_per_m * np.cos(self.theta)]
        )
        self.s1_l_not = self.px_len / 4 * self.scale
        self.x_is_0 = -self.d2 * self.px_per_m * np.sin(self.theta_not) - 1.5*self.s1_l_not
        self.s1_loc = np.asarray(
            [self.x_is_0 + 0.5 * self.s1_l_not + self.x * self.px_per_m,
             self.px_len / 100 * self.scale + self.px_len/8 * self.scale + self.d2 * self.px_per_m * np.cos(self.theta)]
        )
        self.mass_loc = np.asarray(
            [self.x_is_0 + self.x * self.px_per_m,
             self.px_len/8 * self.scale + self.d2 * self.px_per_m * np.cos(self.theta)]
        )

        # Make the spring points
        points = make_spring(height=self.px_len/4, radius=self.px_len/24)

        # Put up a text visual to display time info
        self.font_size = 36.
        self.text = visuals.TextVisual('0:00.00', color='white', pos=[150, 200, 0])
        self.text.font_size = self.font_size

        # Get the pivoting bar ready
        self.rod = visuals.BoxVisual(width=self.px_len/40, height=self.px_len/40, depth=self.px_len, color='white')
        self.rod.transform = transforms.MatrixTransform()
        self.rod.transform.rotate(np.rad2deg(self.theta), (0, 0, 1))
        self.rod.transform.scale((self.scale, self.scale, 0.0001))
        self.rod.transform.translate(self.center - piv_x_y_px)

        # Show the pivote point (optional)
        # self.center_point = visuals.EllipseVisual(center=self.ellipse_center,
        #                                           radius=(self.scale*self.px_len/15, self.scale*self.px_len/15),
        #                                           color='white')

        # Get the upper spring ready.
        self.spring_2 = visuals.TubeVisual(points, radius=self.px_len/100, color=(0.5, 0.5, 1, 1))
        self.spring_2.transform = transforms.MatrixTransform()
        self.spring_2.transform.rotate(90, (0, 1, 0))
        self.spring_2.transform.scale((self.scale, self.scale, 0.0001))
        self.spring_2.transform.translate(self.center + self.s2_loc)

        # Get the lower spring ready.
        self.spring_1 = visuals.TubeVisual(points, radius=self.px_len/100, color=(0.5, 0.5, 1, 1))
        self.spring_1.transform = transforms.MatrixTransform()
        self.spring_1.transform.rotate(90, (0, 1, 0))
        self.spring_1.transform.scale((self.scale * (1.0 - (self.x * self.px_per_m)/(self.scale * self.px_len/2)),
                                       self.scale, 0.0001))
        self.spring_1.transform.translate(self.center + self.s1_loc)

        # Finally, prepare the mass that is being moved
        self.mass = visuals.BoxVisual(width=self.px_len / 4, height=self.px_len / 8,
                                      depth=self.px_len / 4, color='white')
        self.mass.transform = transforms.MatrixTransform()
        self.mass.transform.scale((self.scale, self.scale, 0.0001))
        self.mass.transform.translate(self.center + self.mass_loc)

        # Append all the visuals
        # self.visuals.append(self.center_point)
        self.visuals.append(self.rod)
        self.visuals.append(self.spring_2)
        self.visuals.append(self.spring_1)
        self.visuals.append(self.mass)
        self.visuals.append(self.text)

        # Set up a timer to update the image and give a real-time rendering
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def on_draw(self, ev):
        # Stolen from previous - just clears the screen and redraws stuff
        self.context.set_clear_color((0, 0, 0, 1))
        self.context.set_viewport(0, 0, *self.physical_size)
        self.context.clear()
        for vis in self.visuals:
            vis.draw()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)

        for vis in self.visuals:
            vis.transforms.configure(canvas=self, viewport=vp)

    def on_timer(self, ev):
        # Update x, theta, xdot, thetadot
        self.params_update(dt=1/60)

        # Calculate change for the upper spring, relative to its starting point
        extra_term = self.theta - self.theta_not
        trig_junk = (
            np.sin(self.theta_not) * (np.cos(extra_term) - 1) +
            np.cos(self.theta_not) * np.sin(extra_term)
        )
        delta_x = self.d1 * self.px_per_m * trig_junk

        # Calculate change for the lower spring, relative to something arbitrary so I didn't
        # want to kill myself mathematically
        trig_junk_2 = np.sin(self.theta_not) - np.sin(self.theta)
        first_term = self.d2 * trig_junk_2
        top_term = (first_term - self.x)*self.px_per_m
        net_s1_scales = 1 + top_term/self.s1_l_not
        self.s1_loc[0] = -0.5 * (-self.x * self.px_per_m + self.s1_l_not +
                                 self.d2 * self.px_per_m * (np.sin(self.theta) + np.sin(self.theta_not)))
        self.s1_loc[0] -= 0.5 * net_s1_scales * self.s1_l_not

        # Calculate the new pivot location - this is important because the rotation occurs about
        # the center of the rod, so it has to be offset appropriately
        piv_x_y_px = np.asarray((
            -1*self.pivot_loc_px*np.sin(self.theta),
            self.pivot_loc_px*(np.cos(self.theta) - 1)
        ))

        # Calculate the new mass x location, relative (again) to some simple parameter where x=0
        self.mass_loc[0] = self.x_is_0 + self.x * self.px_per_m

        # Figure out how much time has passed
        millis_passed = int(100 * (self.t % 1))
        sec_passed = int(self.t % 60)
        min_passed = int(self.t // 60)

        # Apply the necessary transformations to the rod
        self.rod.transform.reset()
        self.rod.transform.rotate(np.rad2deg(self.theta), (0, 0, 1))
        self.rod.transform.scale((self.scale, self.scale, 0.0001))
        self.rod.transform.translate(self.center - piv_x_y_px)

        # Redraw and rescale the second spring
        self.spring_2.transform.reset()
        self.spring_2.transform.rotate(90, (0, 1, 0))
        self.spring_2.transform.scale(((1 - (delta_x / (self.scale * self.px_len / 4))) * self.scale,
                                       self.scale,
                                       0.0001))
        self.spring_2.transform.translate(self.center +
                                          self.s2_loc +
                                          np.asarray([delta_x, 0]))

        # Redraw and rescale the second spring (the hardest part to get, mathematically)
        self.spring_1.transform.reset()
        self.spring_1.transform.rotate(90, (0, 1, 0))
        self.spring_1.transform.scale((net_s1_scales * self.scale,
                                       self.scale,
                                       0.0001))
        self.spring_1.transform.translate(self.center +
                                          self.s1_loc)

        # Redrew and rescale the mass
        self.mass.transform.reset()
        self.mass.transform.scale((self.scale, self.scale, 0.0001))
        self.mass.transform.translate(self.center + self.mass_loc)

        # Update the timer with how long it's been
        self.text.text = '{:0>2d}:{:0>2d}.{:0>2d}'.format(min_passed, sec_passed, millis_passed)

        # Trigger all of the drawing and updating
        self.update()

    def params_update(self, dt):
        # Uses Euler update method - Runge-Kutta later.

        # Calculate the second derivative of x
        x_dd_t1 = -self.b * self.x_dot * np.abs(self.x_dot)
        x_dd_t2 = -self.spring_k1*(self.x + self.d2 * self.theta)
        x_dot_dot = (x_dd_t1 + x_dd_t2)/self.little_m

        # Calculate the second derivative of theta
        term1 = -self.spring_k1 * self.d2 * self.x
        term2 = -self.theta * (self.spring_k1*(self.d2 ** 2) + self.spring_k2*(self.d1 ** 2))
        theta_dot_dot = (term1 + term2)/self.j_term

        # Update everything appropriately
        self.t += dt
        self.x += dt*self.x_dot
        self.theta += dt*self.theta_dot
        self.x_dot += dt * x_dot_dot
        self.theta_dot += dt * theta_dot_dot


if __name__ == '__main__':
    win = WigglyBar()
    if sys.flags.interactive != 1:
        win.app.run()
