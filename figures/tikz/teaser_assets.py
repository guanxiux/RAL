"""Raster assets of the intro figure (teaser.tex).  The base image (teaser_assets/base.png, 1536x1024) is cropped to
rows Y0..Y0+640 and cut into a 12x5 grid of 128-px cells, the positions of the schematic convolution; teaser.tex uses
the same grid and the same ACTIVE cells (col, row), the upstream mask's active positions (the pedestrian and the car).
Writes scene.jpg (the cropped frame) and the receptive-field mosaics patches_dense.png (all 60 positions) and
patches_ours.png (the active positions, laid out where they sit in the grid)."""
from PIL import Image, ImageEnhance, ImageOps

CELL, COLS, ROWS, Y0 = 128, 12, 5, 192
ACTIVE = [(4, 1), (4, 2), (4, 3), (5, 2), (6, 2)]
PITCH, PATCH = 132, 120        # mosaic pixels per position and per 3x3 receptive-field patch (2.2 / 2.0 mm in teaser.tex)

base = Image.open('teaser_assets/base.png').convert('RGB')
gray = ImageEnhance.Brightness(ImageOps.grayscale(base)).enhance(1.08).convert('RGB')   # gray, so that the red mask stands out
scene = gray.crop((0, Y0, COLS * CELL, Y0 + ROWS * CELL))
scene.save('teaser_assets/scene.jpg', quality=92)

pad = Image.new('RGB', (scene.width + 2 * CELL, scene.height + 2 * CELL), 'white')   # zero padding at the border
pad.paste(scene, (CELL, CELL))
def patch(c, r):
    return pad.crop((c * CELL, r * CELL, (c + 3) * CELL, (r + 3) * CELL)).resize((PATCH, PATCH), Image.LANCZOS)

def mosaic(cells, name):
    c0, r0 = min(c for c, _ in cells), min(r for _, r in cells)
    cols, rows = max(c for c, _ in cells) - c0 + 1, max(r for _, r in cells) - r0 + 1
    im = Image.new('RGBA', (cols * PITCH, rows * PITCH), (0, 0, 0, 0))
    off = (PITCH - PATCH) // 2
    for c, r in cells:
        im.paste(patch(c, r), ((c - c0) * PITCH + off, (r - r0) * PITCH + off))
    im.save(f'teaser_assets/{name}.png')

mosaic([(c, r) for r in range(ROWS) for c in range(COLS)], 'patches_dense')
mosaic(ACTIVE, 'patches_ours')
