"""Constants for figure generation."""

ALIGN_COLORMAP: str = 'plasma'
"""Color map for alignment tracks."""

COLOR_VARTYPE: dict[str, tuple[float, float, float]] = {
    'INS': (0.25, 0.25, 1.0),
    'DEL': (1.0, 0.25, 0.25),
    'INV': (0.375, 1.0, 0.375),
    'DUP': (1.0, 0.25, 1.0),
    'SNV': (0.0, 0.0, 0.0),
    'RGN': (0.0, 0.0, 0.0),
    'SUB': (0.0, 0.0, 0.0),
}
"""Variant type colors."""

CPX_COLOR_DICT: dict[str, tuple[float, float, float]] = {
    'INS': (0.2510, 0.2510, 1.0000),
    'DEL': (1.0000, 0.2510, 0.2510),
    'INV': (0.3765, 1.0000, 0.3765),

    'DUP': (1.0000, 0.2510, 1.0000),
    'TRP': (0.8510, 0.2118, 0.8510),
    'QUAD': (0.7020, 0.1765, 0.7020),
    'HDUP': (0.3765, 0.1255, 0.3765),

    'INVDUP': (0.3216, 0.8510, 0.3686),
    'INVTRP': (0.2745, 0.7020, 0.2745),
    'INVQUAD': (0.2157, 0.5490, 0.2157),
    'INVHDUP': (0.1569, 0.4000, 0.1569),

    'MIXDUP': (0.8510, 0.5216, 0.1451),
    'MIXTRP': (0.7020, 0.4275, 0.1176),
    'MIXQUAD': (0.5451, 0.3333, 0.0941),
    'MIXHDUP': (0.4000, 0.2471, 0.0706),

    'NML': (0.2000, 0.2000, 0.2000),

    'UNMAPPED': (0.4510, 0.4510, 0.4510),
}
"""Copy number variant colors."""
