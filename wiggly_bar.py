from vispy import app, visuals
from vispy.visuals import transforms
from vispy import scene
import sys
import numpy as np


class WigglyBar(scene.SceneCanvas):
    def __init__(self):
        scene.SceneCanvas.__init__(self, title='Wiggly Bar', size=(800, 800))
        self.unfreeze()
        self.new_vbox = self.central_widget.add_view()
        # self.new_vbox.camera = scene.TurntableCamera(up='z', fov=60)

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
        self.x = -0.01
        self.x_dot = -0.12
        self.theta = 0.005
        self.theta_dot = 0
        self.method = 'Runge-Kutta'
        self.theta_not = self.theta  # I'll need this later

        # Initialize constants for display
        self.px_len = 10
        self.scale = 50
        self.center = np.asarray((500, 450 - 1.5*self.scale, 0.0))
        self.ellipse_center = (500, 450, 0.0)
        self.visuals = []
        self.px_per_m = self.scale * self.px_len/(self.d1 + self.d2)
        self.rod_length = self.scale * self.px_len/self.px_per_m

        # Set up stuff for establishing a pivot point to rotate about
        self.pivot_loc = (self.d1 - self.d2)/2
        self.pivot_loc_px = self.pivot_loc * self.px_per_m
        piv_x_y_px = np.asarray((
            -1*self.pivot_loc_px*np.sin(self.theta),
            self.pivot_loc_px*(1 - np.cos(self.theta)),
            0
        ))

        # Set up positioning info for the springs and mass, as well as some constants for use later
        self.s2_loc = np.asarray(
            [self.d1 * self.px_per_m * np.sin(self.theta) + self.px_len/8 * self.scale,
             self.px_len/8 * self.scale - self.d1 * self.px_per_m * np.cos(self.theta),
             0]
        )
        self.s1_l_not = self.px_len / 4 * self.scale
        self.x_is_0 = -self.d2 * self.px_per_m * np.sin(self.theta_not) - 1.5*self.s1_l_not
        self.s1_loc = np.asarray(
            [self.x_is_0 + self.s1_l_not + 0.5*self.x * self.px_per_m,
             self.px_len/8 * self.scale + self.d2 * self.px_per_m * np.cos(self.theta),
             0]
        )
        self.mass_loc = np.asarray(
            [self.x_is_0 + self.x * self.px_per_m,
             self.px_len/8 * self.scale + self.d2 * self.px_per_m * np.cos(self.theta),
             0]
        )

        # Put up a text visual to display time info
        self.font_size = 36.
        self.text = scene.visuals.Text('0:00.00', color='white', pos=[50, 250, 0],
                                       anchor_x='left', parent=self.new_vbox)
        self.text.font_size = self.font_size

        # Let's put in more text so we know what method is being used to update this
        self.method_text = scene.visuals.Text('Method: {}'.format(self.method),
                                              color='white', pos=[50, 300, 0], anchor_x='left', parent=self.new_vbox)
        self.method_text.font_size = self.font_size - 12.

        # Get the pivoting bar ready
        self.rod = scene.visuals.Box(width=self.px_len/40, height=self.px_len/40,
                                     depth=self.px_len, color='white', parent=self.new_vbox, edge_color='red')
        self.rod.transform = transforms.MatrixTransform()
        # self.rod.transform.matrix *= self.new_vbox.camera.transform.matrix
        self.rod.transform.rotate(np.rad2deg(self.theta), (0, 0, 1))
        self.rod.transform.scale((self.scale, self.scale, 0.00001))
        self.rod.transform.translate(self.center - piv_x_y_px)

        # Show the pivot point (optional)
        # self.center_point = visuals.EllipseVisual(center=self.ellipse_center,
        #                                           radius=(self.scale*self.px_len/15, self.scale*self.px_len/15),
        #                                           color='white')

        # Get the upper spring ready
        self.spring_2 = scene.visuals.Box(width=self.px_len / 4, height=self.px_len / 8,
                                          depth=self.px_len / 8, parent=self.new_vbox, edge_color='red')
        self.spring_2.transform = transforms.MatrixTransform()
        # self.spring_2.transform.matrix *= self.new_vbox.camera.transform.matrix
        self.spring_2.transform.scale((self.scale, self.scale, 0.00001))
        self.spring_2.transform.translate(self.center + self.s2_loc)

        self.spring_1 = scene.visuals.Box(width=self.px_len / 4,
                                          height=self.px_len / 8,
                                          depth=self.px_len / 8,
                                          parent=self.new_vbox,
                                          edge_color='red')
        self.spring_1.transform = transforms.MatrixTransform()
        # self.spring_1.transform.matrix *= self.new_vbox.camera.transform.matrix
        self.spring_1.transform.scale((self.scale * (1.0 - (self.x * self.px_per_m)/(self.scale * self.px_len/4)),
                                       self.scale, 0.00001))
        self.spring_1.transform.translate(self.center + self.s1_loc)

        # Finally, prepare the mass that is being moved
        self.mass = scene.visuals.Box(width=self.px_len / 4, height=self.px_len / 8,
                                      depth=self.px_len / 4, color='white', parent=self.new_vbox, edge_color='red')
        self.mass.transform = transforms.MatrixTransform()
        # self.mass.transform.matrix *= self.new_vbox.camera.transform.matrix
        self.mass.transform.scale((self.scale, self.scale, 0.00001))
        self.mass.transform.translate(self.center + self.mass_loc)

        # centered XYZ axis visual

        # Append all the visuals
        # self.visuals.append(self.center_point)
        self.visuals.append(self.rod)
        self.visuals.append(self.spring_2)
        self.visuals.append(self.spring_1)
        self.visuals.append(self.mass)
        self.visuals.append(self.text)
        self.visuals.append(self.method_text)

        # Set up a timer to update the image and give a real-time rendering
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        for visual in self.visuals:
            try:
                print(visual.transform.matrix)
            except AttributeError:
                continue
        self.freeze()
        self.show()

    def on_timer(self, ev):
        # Update x, theta, xdot, thetadot
        self.params_update(dt=1/60, method=self.method)

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
        self.s1_loc[0] = -0.5 * (-self.x*self.px_per_m + self.s1_l_not +
                                 self.d2*self.px_per_m*(np.sin(self.theta) + np.sin(self.theta_not)))

        # Calculate the new pivot location - this is important because the rotation occurs about
        # the center of the rod, so it has to be offset appropriately
        piv_x_y_px = np.asarray((
            -1*self.pivot_loc_px*np.sin(self.theta),
            self.pivot_loc_px*(np.cos(self.theta) - 1),
            0
        ))

        # Calculate the new mass x location, relative (again) to some simple parameter where x=0
        self.mass_loc[0] = self.x_is_0 + self.x * self.px_per_m

        # Figure out how much time has passed
        millis_passed = int(100 * (self.t % 1))
        sec_passed = int(self.t % 60)
        min_passed = int(self.t // 60)

        # Apply the necessary transformations to the rod
        self.rod.transform.reset()
        # self.rod.transform.matrix *= self.new_vbox.camera.transform.matrix
        self.rod.transform.rotate(np.rad2deg(self.theta), (0, 0, 1))
        self.rod.transform.scale((self.scale, self.scale, 0.00001))
        self.rod.transform.translate(self.center - piv_x_y_px)

        # Redraw and rescale the second spring
        self.spring_2.transform.reset()
        # self.spring_2.transform.matrix *= self.new_vbox.camera.transform.matrix
        self.spring_2.transform.scale(((1 - (delta_x / (self.scale * self.px_len / 4))) * self.scale,
                                       self.scale,
                                       0.00001))
        self.spring_2.transform.translate(self.center +
                                          self.s2_loc +
                                          np.asarray([0.5*delta_x, 0, 0]))

        # Redraw and rescale the second spring (the hardest part to get, mathematically)
        self.spring_1.transform.reset()
        # self.spring_1.transform.matrix *= self.new_vbox.camera.transform.matrix
        self.spring_1.transform.scale((net_s1_scales*self.scale,
                                       self.scale,
                                       0.00001))
        self.spring_1.transform.translate(self.center +
                                          self.s1_loc)

        # Redrew and rescale the mass
        self.mass.transform.reset()
        # self.mass.transform.matrix *= self.new_vbox.camera.transform.matrix
        self.mass.transform.scale((self.scale, self.scale, 0.00001))
        self.mass.transform.translate(self.center + self.mass_loc)

        # Update the timer with how long it's been and the method text
        self.text.text = '{:0>2d}:{:0>2d}.{:0>2d}'.format(min_passed, sec_passed, millis_passed)

        # Trigger all of the drawing and updating
        self._update_transforms()

        self.update()

    def params_update(self, dt, method='euler'):
        # Uses either Euler method or Runge-Kutta, depending on your input to "method"

        if method.lower() == 'euler':
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
        elif method.lower() == 'runge-kutta':
            self._runge_kutta_update(dt)

    def _runge_kutta_update(self, dt):
        info_vector = np.asarray([self.x_dot, self.theta_dot, self.x, self.theta]).copy()

        k1 = [((-self.b * info_vector[0] * np.abs(info_vector[0]))+
               (-self.spring_k1*(info_vector[2] + self.d2 * info_vector[3])))/self.little_m,
              ((-self.spring_k1 * self.d2 * info_vector[2]) +
               (-info_vector[3] * (self.spring_k1*(self.d2 ** 2) + self.spring_k2*(self.d1 ** 2))))/self.j_term,
              info_vector[0],
              info_vector[1]]

        k1 = np.asarray(k1) * dt

        updated_est = info_vector + 0.5 * k1

        k2 = [((-self.b * updated_est[0] * np.abs(updated_est[0]))+
               (-self.spring_k1*(updated_est[2] + self.d2 * updated_est[3])))/self.little_m,
              ((-self.spring_k1 * self.d2 * updated_est[2]) +
               (-updated_est[3] * (self.spring_k1*(self.d2 ** 2) + self.spring_k2*(self.d1 ** 2))))/self.j_term,
              updated_est[0],
              updated_est[1]]

        k2 = np.asarray(k2) * dt

        updated_est = info_vector - k1 + 2 * k2

        k3 = [((-self.b * updated_est[0] * np.abs(updated_est[0]))+
               (-self.spring_k1*(updated_est[2] + self.d2 * updated_est[3])))/self.little_m,
              ((-self.spring_k1 * self.d2 * updated_est[2]) +
               (-updated_est[3] * (self.spring_k1*(self.d2 ** 2) + self.spring_k2*(self.d1 ** 2))))/self.j_term,
              updated_est[0],
              updated_est[1]]

        k3 = np.asarray(k3) * dt

        final_est = info_vector + (1/6)*(k1 + 4*k2 + k3)

        self.x_dot, self.theta_dot, self.x, self.theta = final_est.copy()
        self.t += dt


if __name__ == '__main__':
    win = WigglyBar()
    if sys.flags.interactive != 1:
        win.app.run()
