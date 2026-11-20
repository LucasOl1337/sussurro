"""Icones de linha da barra lateral, desenhados em PIL (sem arquivo de imagem).

Grade de 24 unidades, traco de ~1,7 px com pontas redondas, renderizado em 4x e
reduzido: fica nitido em 100% e 200% de escala.
"""
from PIL import Image, ImageDraw

SS = 4
GRID = 24.0


class _Pen:
    def __init__(self, size: int, color: str, stroke: float):
        self.px = size * SS
        self.k = self.px / GRID
        self.img = Image.new("RGBA", (self.px, self.px), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.color = color
        self.w = max(1, round(stroke * SS))

    def p(self, x, y):
        return x * self.k, y * self.k

    def dot(self, x, y, r=None):
        r = (self.w / 2) if r is None else r * self.k
        cx, cy = self.p(x, y)
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=self.color)

    def line(self, *pts):
        xy = [self.p(x, y) for x, y in pts]
        self.d.line(xy, fill=self.color, width=self.w, joint="curve")
        for x, y in (pts[0], pts[-1]):
            self.dot(x, y)

    def rrect(self, x0, y0, x1, y1, r):
        self.d.rounded_rectangle([*self.p(x0, y0), *self.p(x1, y1)], radius=r * self.k,
                                 outline=self.color, width=self.w)

    def circle(self, x, y, r, fill=False):
        box = [*self.p(x - r, y - r), *self.p(x + r, y + r)]
        if fill:
            self.d.ellipse(box, fill=self.color)
        else:
            self.d.ellipse(box, outline=self.color, width=self.w)

    def arc(self, x0, y0, x1, y1, start, end):
        self.d.arc([*self.p(x0, y0), *self.p(x1, y1)], start, end, fill=self.color, width=self.w)

    def done(self, size):
        return self.img.resize((size, size), Image.LANCZOS)


def _mic(p):
    p.rrect(9, 2.5, 15, 14, 3)
    p.arc(5, 4, 19, 18, 0, 180)
    p.line((12, 18), (12, 21.5))


def _wave(p):
    for x, h in ((4, 3), (8, 8), (12, 15), (16, 8), (20, 3)):
        p.line((x, 12 - h / 2), (x, 12 + h / 2))


def _record(p):
    p.circle(12, 12, 9)
    p.circle(12, 12, 4, fill=True)


def _book(p):
    p.rrect(5, 2.5, 19, 21.5, 2.5)
    p.line((9, 2.8), (9, 21.2))
    p.line((12, 8), (16, 8))
    p.line((12, 12), (15, 12))


def _chart(p):
    p.line((6, 20), (6, 13))
    p.line((12, 20), (12, 5))
    p.line((18, 20), (18, 10))


def _split(p):
    p.rrect(3, 4.5, 21, 19.5, 2.5)
    p.line((12, 4.8), (12, 19.2))


def _headset(p):
    p.arc(4, 4, 20, 20, 180, 360)
    p.line((4, 12), (4, 14))
    p.line((20, 12), (20, 14))
    p.rrect(3, 13, 8, 20, 1.8)
    p.rrect(16, 13, 21, 20, 1.8)


def _sliders(p):
    for y, kx in ((6, 9), (12, 16), (18, 7)):
        p.line((3.5, y), (kx - 2.4, y))
        p.line((kx + 2.4, y), (20.5, y))
        p.circle(kx, y, 2.2)


def _search(p):
    p.circle(10.5, 10.5, 6.5)
    p.line((15.5, 15.5), (20.5, 20.5))


def _play(p):
    p.d.polygon([p.p(8, 5), p.p(8, 19), p.p(19, 12)], fill=p.color)


def _copy(p):
    p.rrect(8, 8, 20, 20, 2.5)
    p.line((4.5, 15), (4.5, 6.5))
    p.arc(4.5 - 0.0, 4, 9.5, 9, 180, 270)
    p.line((7, 4), (15, 4))


ICONS = {
    "historico": _mic, "aovivo": _wave, "reuniao": _record, "biblioteca": _book,
    "estatisticas": _chart, "comparar": _split, "omarchy": _headset, "config": _sliders,
    "busca": _search, "play": _play, "copy": _copy,
}


def icon_image(name: str, color: str = "#1a1a1a", size: int = 18, stroke: float = 1.7):
    """PIL RGBA do icone `name` no tamanho final (`size` px)."""
    pen = _Pen(size, color, stroke)
    ICONS[name](pen)
    return pen.done(size)


def brand_image(color: str = "#1a1a1a", height: int = 22):
    """Marca do Sussurro: sete barras de onda, mais alta no meio."""
    heights = (5, 10, 16, 22, 16, 10, 5)
    bar, gap = 2.6, 2.4
    w = len(heights) * bar + (len(heights) - 1) * gap
    k = SS * height / 22.0
    img = Image.new("RGBA", (round(w * k), round(22 * k)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, h in enumerate(heights):
        x0 = i * (bar + gap) * k
        y0 = (22 - h) / 2 * k
        d.rounded_rectangle([x0, y0, x0 + bar * k, y0 + h * k], radius=bar * k / 2, fill=color)
    return img.resize((round(w * height / 22.0), height), Image.LANCZOS)
