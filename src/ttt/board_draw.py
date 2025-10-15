import random

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

_CANVAS_SIZE = 512
_FINAL_SIZE = 256


def _add_perspective(
    image: Image.Image, fillcolor: tuple[int, int, int]
) -> Image.Image:
    canvas_scale = random.uniform(1.2, 2.0)
    width, height = image.size

    # fmt: off
    perspective_matrix = [
        1, random.uniform(-0.2, 0.2), -(canvas_scale - 1) * width / 2,
        random.uniform(-0.2, 0.2), 1, -(canvas_scale - 1) * height / 2,
        0, 0, 1,
    ]
    # fmt: on

    width, height = [int(x * canvas_scale) for x in (width, height)]
    image = image.transform(
        (width, height),
        Image.Transform.PERSPECTIVE,
        perspective_matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fillcolor,
    )
    return image


def to_image(board_str: list[str]) -> Image.Image:
    img_size = _CANVAS_SIZE
    board_size = 3
    line_thickness = int(5 / 300 * img_size)
    bg_color = tuple(random.randint(192, 255) for _ in range(3))
    assert len(bg_color) == 3  # Hint linter that size is indeed 3.
    image = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(image)

    cell_size = img_size // board_size
    line_color = tuple(random.randint(0, 92) for _ in range(3))
    x_colors = [(0, 0, 0), (128, 64, 0), (0, 64, 128), (92, 64, 51), (60, 93, 52)]
    o_colors = x_colors
    x_color = random.choice(x_colors)
    o_color = random.choice(o_colors)

    # Draw the baord grids.
    for i in range(1, board_size):
        deviations = [random.randint(-5, 5) for _ in range(4)]
        draw.line(
            (i * cell_size + deviations[0], 0, i * cell_size + deviations[1], img_size),
            fill=line_color,
            width=line_thickness,
        )
        draw.line(
            (0, i * cell_size + deviations[2], img_size, i * cell_size + deviations[3]),
            fill=line_color,
            width=line_thickness,
        )

    # Draw the characters.
    for row in range(board_size):
        for col in range(board_size):
            index = row * board_size + col
            char = board_str[index]
            if char != ".":
                # # Simple draw -
                # text_x = col * cell_size + cell_size // 2
                # text_y = row * cell_size + cell_size // 2
                # font = ImageFont.load_default()
                # draw.text((text_x, text_y), char, fill='black', font=font, anchor="mm")

                font_size = int(
                    random.uniform(36 / 300 * img_size, 55 / 300 * img_size)
                )
                font = ImageFont.truetype(
                    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                    size=font_size,
                )

                text_image = Image.new("RGBA", (cell_size, cell_size), bg_color + (0,))
                text_draw = ImageDraw.Draw(text_image)
                textbbox = text_draw.textbbox((0, 0), char, font)
                text_color = x_color if char == "X" else o_color
                text_draw.text((0, 0), char, fill=text_color, font=font)
                text_x_offset = int(
                    col * cell_size + cell_size // 2 - textbbox[0] - textbbox[2] // 2
                )
                text_y_offset = int(
                    row * cell_size + cell_size // 2 - textbbox[1] - textbbox[3] // 2
                )
                degrees = random.randint(-10, 10)
                text_image = text_image.rotate(
                    degrees,
                    center=(
                        (textbbox[0] + textbbox[2]) // 2,
                        (textbbox[1] + textbbox[3]) // 2,
                    ),
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                    fillcolor=bg_color + (0,),
                )
                image.paste(text_image, (text_x_offset, text_y_offset), text_image)
    image = _add_perspective(image, fillcolor=bg_color)
    image = image.resize((_FINAL_SIZE, _FINAL_SIZE), resample=Image.Resampling.BICUBIC)
    return image
