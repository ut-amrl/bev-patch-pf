import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from scipy.spatial.transform import Rotation as R

from geotiff.handler import GeoTiffHandler


def align_trajectory_gui(geotiff_path: str, traj_path: str):
    geo_handler = GeoTiffHandler(geotiff_path)
    timestamp, traj_xyr = load_trajectory(traj_path)
    transformed_df = [None]

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.45)
    ax.imshow(geo_handler.image)
    traj_scatter = ax.scatter([], [], c="red", s=1)
    traj_quiver = [None]

    min_x, min_y, max_x, max_y = geo_handler.bounds
    init_x = (min_x + max_x) / 2
    init_y = (min_y + max_y) / 2
    init_r = 0.0

    # sliders
    axcolor = "lightgoldenrodyellow"
    ax_x = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
    ax_r = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    s_x = Slider(ax_x, "Offset X", min_x - 100, max_x + 100, valinit=init_x, valfmt="%.2f")
    s_y = Slider(ax_y, "Offset Y", min_y - 100, max_y + 100, valinit=init_y, valfmt="%.2f")
    s_r = Slider(ax_r, "Rotation (rad)", -np.pi, np.pi, valinit=init_r, valfmt="%.3f")

    # text boxes
    text_x = TextBox(plt.axes([0.25, 0.18, 0.1, 0.035]), "X Val", initial=f"{init_x:.3f}")
    text_y = TextBox(plt.axes([0.40, 0.18, 0.1, 0.035]), "Y Val", initial=f"{init_y:.3f}")
    text_r = TextBox(plt.axes([0.55, 0.18, 0.1, 0.035]), "R Val", initial=f"{init_r:.4f}")

    def transform_and_plot(offset_x, offset_y, offset_r):
        transformed = apply_offset(traj_xyr, offset_x, offset_y, offset_r)
        traj_uvr = geo_handler.coords_to_uvr(transformed)

        transformed_df[0] = pd.DataFrame(
            {
                "timestamp": timestamp,
                "x": transformed[:, 0],
                "y": transformed[:, 1],
                "angle": transformed[:, 2],
            }
        )

        traj_scatter.set_offsets(traj_uvr[:, :2])

        step = 100
        quiver_indices = np.arange(0, len(traj_uvr), step)
        if len(quiver_indices) == 0:
            return

        x = traj_uvr[quiver_indices, 0]
        y = traj_uvr[quiver_indices, 1]
        theta = traj_uvr[quiver_indices, 2]

        u = np.cos(-theta) * 20
        v = np.sin(-theta) * 20

        if traj_quiver[0] is not None:
            traj_quiver[0].remove()
        traj_quiver[0] = ax.quiver(x, y, u, v, angles="xy", scale_units="xy", scale=0.5, color="b")
        fig.canvas.draw_idle()

    def update(val):
        transform_and_plot(s_x.val, s_y.val, s_r.val)
        update_textboxes(None)

    def update_textboxes(_):
        text_x.set_val(f"{s_x.val:.3f}")
        text_y.set_val(f"{s_y.val:.3f}")
        text_r.set_val(f"{s_r.val:.4f}")

    def make_submit_fn(slider):
        return lambda text: slider.set_val(float(text)) if text else None

    # Connect sliders and text boxes
    s_x.on_changed(update)
    s_y.on_changed(update)
    s_r.on_changed(update)
    s_x.on_changed(update_textboxes)
    s_y.on_changed(update_textboxes)
    s_r.on_changed(update_textboxes)

    text_x.on_submit(make_submit_fn(s_x))
    text_y.on_submit(make_submit_fn(s_y))
    text_r.on_submit(make_submit_fn(s_r))

    # Save button
    def on_save(event):
        outfile = traj_path.replace(".csv", "_aligned.csv")
        if transformed_df[0] is not None:
            df = transformed_df[0]
            df_formatted = pd.DataFrame(
                {
                    "timestamp": df["timestamp"].map(lambda x: f"{x:.6f}"),
                    "x": df["x"].map(lambda x: f"{x:.4f}"),
                    "y": df["y"].map(lambda x: f"{x:.4f}"),
                    "angle": df["angle"].map(lambda x: f"{x:.4f}"),
                }
            )
            df_formatted.to_csv(outfile, index=False)
            print(f"Aligned trajectory saved to {outfile}")
        else:
            print("No transformed trajectory to save.")

    ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(ax_button, "Save")
    button.on_clicked(on_save)

    transform_and_plot(init_x, init_y, init_r)
    plt.show()


def load_trajectory(traj_path: str):
    df = pd.read_csv(traj_path)
    quats = df[["qx", "qy", "qz", "qw"]].values
    rot_mats = R.from_quat(quats).as_matrix()
    yaws = np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0])
    return df["timestamp"].values, np.column_stack([df["x"], df["y"], yaws])


def apply_offset(traj_xyr, offset_x, offset_y, offset_r):
    c, s = np.cos(offset_r), np.sin(offset_r)
    x = c * traj_xyr[:, 0] - s * traj_xyr[:, 1] + offset_x
    y = s * traj_xyr[:, 0] + c * traj_xyr[:, 1] + offset_y
    r = traj_xyr[:, 2] + offset_r
    return np.column_stack([x, y, r])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align trajectory with GeoTIFF (GUI).")
    parser.add_argument("--geotiff", type=str, required=True, help="Path to GeoTIFF file")
    parser.add_argument("--traj", type=str, required=True, help="Path to trajectory CSV file")
    args = parser.parse_args()

    align_trajectory_gui(args.geotiff, args.traj)
