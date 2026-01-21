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
    'CPX': (0.0, 0.0, 0.0),

    'CPX_INS': (0.2510, 0.2510, 1.0000),
    'CPX_DEL': (1.0000, 0.2510, 0.2510),
    'CPX_INV': (0.3765, 1.0000, 0.3765),

    'CPX_DUP': (1.0000, 0.2510, 1.0000),
    'CPX_TRP': (0.8510, 0.2118, 0.8510),
    'CPX_QUAD': (0.7020, 0.1765, 0.7020),
    'CPX_HDUP': (0.3765, 0.1255, 0.3765),

    'CPX_INVDUP': (0.3216, 0.8510, 0.3686),
    'CPX_INVTRP': (0.2745, 0.7020, 0.2745),
    'CPX_INVQUAD': (0.2157, 0.5490, 0.2157),
    'CPX_INVHDUP': (0.1569, 0.4000, 0.1569),

    'CPX_MIXDUP': (0.8510, 0.5216, 0.1451),
    'CPX_MIXTRP': (0.7020, 0.4275, 0.1176),
    'CPX_MIXQUAD': (0.5451, 0.3333, 0.0941),
    'CPX_MIXHDUP': (0.4000, 0.2471, 0.0706),

    'CPX_NML': (0.2000, 0.2000, 0.2000),

    'CPX_UNMAPPED': (0.4510, 0.4510, 0.4510),
}
"""Variant type colors."""
