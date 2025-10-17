import random

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pydantic import BaseModel

from typing import Optional

_BOARD_SIZE = 3

_CANVAS_SIZE = 512
_FINAL_SIZE = 256


class RenderParams(BaseModel):
    """Configuration to render a board."""

    # Board-level random parameters
    bg_color: tuple[int, int, int]
    line_color: tuple[int, int, int]
    x_color: tuple[int, int, int]
    o_color: tuple[int, int, int]
    # 4 deviations (horizontal and vertical combined) for _BOARD_SIZE - 1 lines.
    grid_deviations: list[list[int]]
    font_sizes: list[int]  # 9 cells
    text_rotations: list[int]  # 9 cells
    perspective_scale: float
    perspective_mat: list[float]

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "RenderParams":
        rnd = random.Random(seed)
        bg_color = tuple(rnd.randint(192, 255) for _ in range(3))
        assert len(bg_color) == 3  # Hint linter that size is indeed 3.
        line_color = tuple(rnd.randint(0, 92) for _ in range(3))
        assert len(line_color) == 3  # Hint linter that size is indeed 3.

        x_colors = [(0, 0, 0), (128, 64, 0), (0, 64, 128), (92, 64, 51), (60, 93, 52)]
        o_colors = x_colors
        x_color = x_colors[rnd.randint(0, len(x_colors) - 1)]
        o_color = o_colors[rnd.randint(0, len(o_colors) - 1)]
        grid_deviations = [
            [rnd.randint(-5, 5) for _ in range(4)] for _ in range(_BOARD_SIZE - 1)
        ]
        font_sizes = [
            int(rnd.uniform(36 / 300 * 512, 55 / 300 * 512)) for _ in range(9)
        ]
        text_rotations = [rnd.randint(-10, 10) for _ in range(9)]

        canvas_scale = rnd.uniform(1.5, 2.0)  # Perspective scale.

        # Note: Elemetns [2] and [5] depend on width and will be computed on the fly.
        # fmt: off
        perspective_mat = [
            1, rnd.uniform(-0.2, 0.2), 0,
            rnd.uniform(-0.2, 0.2), 1, 0,
            0, rnd.uniform(0, 0.0004), 1,
        ]
        # fmt: on
        return cls(
            bg_color=bg_color,
            line_color=line_color,
            x_color=x_color,
            o_color=o_color,
            grid_deviations=grid_deviations,
            font_sizes=font_sizes,
            text_rotations=text_rotations,
            perspective_scale=canvas_scale,
            perspective_mat=perspective_mat,
        )


class _BoardRenderer:
    def __init__(self, render_params: RenderParams) -> None:
        self._rp = render_params

    def _add_perspective(
        self, image: Image.Image, fillcolor: tuple[int, int, int]
    ) -> Image.Image:
        canvas_scale = self._rp.perspective_scale
        width, height = image.size

        # fmt: off
        perspective_matrix = self._rp.perspective_mat.copy()
        perspective_matrix[2] = -(canvas_scale - 1) * width / 2
        perspective_matrix[5] = -(canvas_scale - 1) * height / 2
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

    def render_board(self, board_str: list[str]) -> Image.Image:
        img_size = _CANVAS_SIZE
        line_thickness = int(5 / 300 * img_size)
        image = Image.new("RGB", (img_size, img_size), self._rp.bg_color)
        draw = ImageDraw.Draw(image)

        cell_size = img_size // _BOARD_SIZE

        # Draw the baord grids.
        for i in range(1, _BOARD_SIZE):
            draw.line(
                (
                    i * cell_size + self._rp.grid_deviations[i - 1][0],
                    0,
                    i * cell_size + self._rp.grid_deviations[i - 1][1],
                    img_size,
                ),
                fill=self._rp.line_color,
                width=line_thickness,
            )
            draw.line(
                (
                    0,
                    i * cell_size + self._rp.grid_deviations[i - 1][2],
                    img_size,
                    i * cell_size + self._rp.grid_deviations[i - 1][3],
                ),
                fill=self._rp.line_color,
                width=line_thickness,
            )

        # Draw the characters.
        for row in range(_BOARD_SIZE):
            for col in range(_BOARD_SIZE):
                index = row * _BOARD_SIZE + col
                char = board_str[index]
                if char != ".":
                    # # Simple draw -
                    # text_x = col * cell_size + cell_size // 2
                    # text_y = row * cell_size + cell_size // 2
                    # font = ImageFont.load_default()
                    # draw.text((text_x, text_y), char, fill='black', font=font, anchor="mm")

                    font = ImageFont.truetype(
                        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                        size=self._rp.font_sizes[index],
                    )

                    text_image = Image.new(
                        "RGBA", (cell_size, cell_size), self._rp.bg_color + (0,)
                    )
                    text_draw = ImageDraw.Draw(text_image)
                    textbbox = text_draw.textbbox((0, 0), char, font)
                    text_color = self._rp.x_color if char == "X" else self._rp.o_color
                    text_draw.text((0, 0), char, fill=text_color, font=font)
                    text_x_offset = int(
                        col * cell_size
                        + cell_size // 2
                        - textbbox[0]
                        - textbbox[2] // 2
                    )
                    text_y_offset = int(
                        row * cell_size
                        + cell_size // 2
                        - textbbox[1]
                        - textbbox[3] // 2
                    )
                    degrees = self._rp.text_rotations[index]
                    text_image = text_image.rotate(
                        degrees,
                        center=(
                            (textbbox[0] + textbbox[2]) // 2,
                            (textbbox[1] + textbbox[3]) // 2,
                        ),
                        resample=Image.Resampling.BICUBIC,
                        expand=True,
                        fillcolor=self._rp.bg_color + (0,),
                    )
                    image.paste(text_image, (text_x_offset, text_y_offset), text_image)
        image = self._add_perspective(image, fillcolor=self._rp.bg_color)
        image = image.resize(
            (_FINAL_SIZE, _FINAL_SIZE), resample=Image.Resampling.BICUBIC
        )
        return image


def to_image(board_str: list[str], render_params: RenderParams) -> Image.Image:
    return _BoardRenderer(render_params).render_board(board_str)
