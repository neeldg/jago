from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

he_dir = Path("figures/he_patches_named")
graph_dir = Path("figures/ranked_patches_clean")
out_path = Path("figures/he_graph_pair_panel.png")

items = [
    {
        "label": "A. Tumor-rich",
        "he": "tumor_rich_he.png",
        "graph": "tumor_rich_patch_000042.png",
    },
    {
        "label": "B. Stromal-rich",
        "he": "stromal_rich_he.png",
        "graph": "stromal_rich_patch_000049.png",
    },
    {
        "label": "C. Immune-rich",
        "he": "immune_rich_he.png",
        "graph": "immune_rich_patch_000002.png",
    },
    {
        "label": "D. Tumor-immune contact",
        "he": "tumor_immune_contact_he.png",
        "graph": "tumor_immune_contact_patch_000015.png",
    },
    {
        "label": "E. Tumor-stroma contact",
        "he": "tumor_stroma_contact_he.png",
        "graph": "tumor_stroma_contact_patch_000043.png",
    },
    {
        "label": "F. Mixed architecture",
        "he": "mixed_architecture_he.png",
        "graph": "mixed_architecture_patch_000034.png",
    },
]

image_size = 360
label_height = 42
header_height = 42
pad = 18
pair_gap = 12

rows = len(items)
cols = 2

try:
    title_font = ImageFont.truetype("Arial.ttf", 26)
    label_font = ImageFont.truetype("Arial.ttf", 22)
    small_font = ImageFont.truetype("Arial.ttf", 20)
except Exception:
    title_font = ImageFont.load_default()
    label_font = ImageFont.load_default()
    small_font = ImageFont.load_default()


def load_square(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img


panel_w = pad * 3 + image_size * 2 + pair_gap
panel_h = pad * 2 + header_height + rows * (label_height + image_size + pad)

canvas = Image.new("RGB", (panel_w, panel_h), "white")
draw = ImageDraw.Draw(canvas)

# Column headers
x_he = pad
x_graph = pad + image_size + pair_gap
y_header = pad

draw.text((x_he, y_header), "H&E crop", fill="black", font=title_font)
draw.text((x_graph, y_header), "JAGO graph", fill="black", font=title_font)

y = pad + header_height

for item in items:
    label = item["label"]

    he_path = he_dir / item["he"]
    graph_path = graph_dir / item["graph"]

    if not he_path.exists():
        raise FileNotFoundError(f"Missing H&E image: {he_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph image: {graph_path}")

    draw.text((pad, y), label, fill="black", font=label_font)
    y_img = y + label_height

    he_img = load_square(he_path, image_size)
    graph_img = load_square(graph_path, image_size)

    canvas.paste(he_img, (x_he, y_img))
    canvas.paste(graph_img, (x_graph, y_img))

    y = y_img + image_size + pad

out_path.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out_path, dpi=(300, 300))
print(f"Saved {out_path}")
