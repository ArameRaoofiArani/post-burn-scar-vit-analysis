"""Step 0 -- write the annotated tiles back into the QuPath project and
render a check thumbnail per slide.

    python 01_write_tiles_to_qupath.py

For every image in the QuPath project this traces its point annotations into
a closed outline, fills that outline on the 224 px tile grid, and adds one
QuPath tile object per enclosed tile. It then saves a downsampled thumbnail
of the slide with those tiles drawn on in red, so the traced region can be
eyeballed against the actual tissue before anything is scored against it.

WARNING: this opens the QuPath project in append mode and clears each
image's existing detections before adding the new tiles. It is the only
script here that modifies the project. Back the project up before the first
run.

Requires OpenSlide and a QuPath installation -- see README.md.
"""

import os

import config

# OpenSlide's DLLs (Windows) and QuPath must be locatable before the imports
# below, so these two calls have to come first.
config.configure_openslide()
config.configure_qupath()

import pandas as pd                                         # noqa: E402
from shapely.geometry import Polygon                        # noqa: E402
from openslide import OpenSlide                             # noqa: E402
from PIL import ImageDraw                                   # noqa: E402
from paquo.projects import QuPathProject                    # noqa: E402

from annotations import (                                   # noqa: E402
    order_points_annotation,
    two_opt_cleanup,
    get_tiles_inside_boundary,
)
from config import (                                        # noqa: E402
    SLIDE_DIR,
    OUTPUT_DIR,
    QUPATH_PROJECT,
    TILE_SIZE as tile_size,
)

THUMBNAIL_MAX_WIDTH = 2000


def main():
    config.ensure_dirs(OUTPUT_DIR / "annotation_thumbnails")

    with QuPathProject(str(QUPATH_PROJECT), mode='a') as project:
        for image in project.images:
            name = image.image_name
            image.hierarchy.detections.clear()
            print(f"\n{'='*60}\n{name}")

            records = []
            for annotation in image.hierarchy.annotations:
                roi = annotation.roi
                if roi.geom_type == "MultiPoint":
                    for point in roi.geoms:
                        records.append({"x_px": point.x, "y_px": point.y})

            df = pd.DataFrame(records)
            print(f"  Points: {len(df)}")

            if len(df) < 3:
                print("  Not enough points, skipping.")
                continue

            ordered = order_points_annotation(list(zip(df["x_px"], df["y_px"])))
            ordered = two_opt_cleanup(ordered, verbose_name=name)
            tiles = get_tiles_inside_boundary(ordered, tile_size=tile_size)
            print(f"  Tiles extracted     : {len(tiles)}")

            for t in tiles:
                image.hierarchy.add_tile(
                    roi=Polygon.from_bounds(
                        t["tile_x"],             t["tile_y"],
                        t["tile_x"] + tile_size, t["tile_y"] + tile_size,
                    )
                )

            # Thumbnail
            slide = OpenSlide(os.path.join(str(SLIDE_DIR), image.image_name))
            width, height = slide.dimensions
            bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
            bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
            thumbnail = slide.get_thumbnail(
                (THUMBNAIL_MAX_WIDTH, int(height / width * THUMBNAIL_MAX_WIDTH)))
            actual_w, actual_h = thumbnail.size
            scale_x, scale_y = actual_w / width, actual_h / height

            draw = ImageDraw.Draw(thumbnail)
            for t in tiles:
                x0 = (t["tile_x"] + bounds_x) * scale_x
                y0 = (t["tile_y"] + bounds_y) * scale_y
                draw.rectangle([x0, y0, x0 + tile_size * scale_x, y0 + tile_size * scale_y],
                               outline="red", width=2)

            out = OUTPUT_DIR / "annotation_thumbnails" / f"{name}_thumbnail.png"
            thumbnail.save(out)
            print(f"  Thumbnail saved     : {out}")
            slide.close()


if __name__ == "__main__":
    main()
