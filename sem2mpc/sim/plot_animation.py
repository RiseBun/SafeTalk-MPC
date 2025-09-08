import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

def plot_trajectory_animation(xs, save_path="mpc_animation.mp4", obstacle=(1.0,0.5,0.35), fps=10):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1, 4); ax.set_ylim(-1, 4); ax.set_aspect('equal'); ax.grid(True)
    xs = np.asarray(xs)
    x_vals, y_vals = xs[:,0], xs[:,1]

    # static artists
    line, = ax.plot([], [], 'b-', lw=2, label='traj')
    point, = ax.plot([], [], 'ro', ms=6)
    obs = Circle((obstacle[0], obstacle[1]), obstacle[2], color='gray', alpha=0.35)
    ax.add_patch(obs)

    # proximity check
    dists = np.sqrt((x_vals - obstacle[0])**2 + (y_vals - obstacle[1])**2)
    min_dist = float(np.min(dists))
    if min_dist < obstacle[2]:
        idx = int(np.argmin(dists))
        ax.plot([x_vals[idx]],[y_vals[idx]], 'kx', ms=10, label='closest')
        ax.legend()
        print(f"⚠️ Min dist {min_dist:.3f} < radius {obstacle[2]:.3f} (possible violation)")

    def init():
        line.set_data([], []); point.set_data([], []); return line, point
    def update(i):
        line.set_data(x_vals[:i+1], y_vals[:i+1])
        point.set_data([x_vals[i]], [y_vals[i]])
        return line, point

    ani = animation.FuncAnimation(fig, update, frames=len(x_vals), init_func=init, blit=True,
                                  interval=int(1000/max(fps,1)))

    try:
        ani.save(save_path, writer='ffmpeg', fps=fps)
        print(f"🎥 saved {save_path}")
    except Exception:
        gif = save_path.replace('.mp4','.gif')
        ani.save(gif, writer='pillow', fps=fps)
        print(f"🎞️ ffmpeg unavailable -> saved {gif}")
    finally:
        plt.close(fig)

    return min_dist
