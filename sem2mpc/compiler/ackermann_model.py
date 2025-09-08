import casadi as ca

class AckermannModel:
    """Simple Ackermann kinematic bicycle model.
    State:  x, y, theta, v, delta
    Input:  a (accel), delta_dot (steer rate)
    Params: L (wheelbase)
    """
    def __init__(self, L: float = 0.5):
        self.L = L
        self.nx = 5
        self.nu = 2

    def forward(self, x, u, dt):
        x_pos, y_pos, theta, v, delta = x[0], x[1], x[2], x[3], x[4]
        a, delta_dot = u[0], u[1]

        # Kinematic bicycle (no slip)
        x_next = x_pos + v * ca.cos(theta) * dt
        y_next = y_pos + v * ca.sin(theta) * dt
        theta_next = theta + v * ca.tan(delta) / self.L * dt
        v_next = v + a * dt
        delta_next = delta + delta_dot * dt

        return ca.vertcat(x_next, y_next, theta_next, v_next, delta_next)
