from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# ============================================================
# À MODIFIER
# ============================================================
before_path = Path("figures/evaluation/feedback_before.png")
after_path  = Path("figures/evaluation/feedback_after.png")
out_path    = Path("figures/evaluation/feedback_before_after.png")

title_top = "Avant feedback local"
title_bottom = "Après feedback local"

# ============================================================
# PARAMÈTRES DE MISE EN PAGE
# ============================================================
margin = 40
gap_between_blocks = 50
title_height = 50
background_color = "white"
text_color = "black"
separator_color = (180, 180, 180)

# ============================================================
# CHARGEMENT DES IMAGES
# ============================================================
img1 = Image.open(before_path).convert("RGB")
img2 = Image.open(after_path).convert("RGB")

# ============================================================
# METTRE LES 2 IMAGES À LA MÊME LARGEUR
# ============================================================
target_width = max(img1.width, img2.width)

def resize_to_width(img, width):
    if img.width == width:
        return img
    new_height = int(img.height * width / img.width)
    return img.resize((width, new_height), Image.LANCZOS)

img1 = resize_to_width(img1, target_width)
img2 = resize_to_width(img2, target_width)

# ============================================================
# POLICE
# ============================================================
try:
    font = ImageFont.truetype("arial.ttf", 28)
except:
    font = ImageFont.load_default()

# ============================================================
# CALCUL DE LA TAILLE DU CANVAS FINAL
# ============================================================
canvas_width = target_width + 2 * margin
canvas_height = (
    margin
    + title_height + img1.height
    + gap_between_blocks
    + title_height + img2.height
    + margin
)

canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)
draw = ImageDraw.Draw(canvas)

# ============================================================
# BLOC 1 : AVANT
# ============================================================
y = margin

draw.text((margin, y), title_top, fill=text_color, font=font)
y += title_height

canvas.paste(img1, (margin, y))
y += img1.height

# ligne de séparation
sep_y = y + gap_between_blocks // 2
draw.line((margin, sep_y, canvas_width - margin, sep_y), fill=separator_color, width=2)

y += gap_between_blocks

# ============================================================
# BLOC 2 : APRÈS
# ============================================================
draw.text((margin, y), title_bottom, fill=text_color, font=font)
y += title_height

canvas.paste(img2, (margin, y))

# ============================================================
# SAUVEGARDE
# ============================================================
out_path.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out_path)

print(f"Figure sauvegardée dans : {out_path}")