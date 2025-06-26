import imageio
import os
from natsort import natsorted  # To sort filenames numerically

def make_gif_from_frames(frame_folder="frames", gif_name="simulation.gif", fps=10):
    # Get all PNG files and sort them numerically
    filenames = natsorted([
        os.path.join(frame_folder, fname)
        for fname in os.listdir(frame_folder)
        if fname.endswith(".png")
    ])

    if not filenames:
        print("❌ No frames found in folder.")
        return

    # Load images and write to GIF
    images = [imageio.imread(fname) for fname in filenames]
    imageio.mimsave(gif_name, images, fps=fps)

    print(f"✅ GIF saved as '{gif_name}' with {len(images)} frames at {fps} fps.")


make_gif_from_frames(frame_folder="frames_b", gif_name="density_evolution_512.gif", fps=10)