"""PAV documentation configuration."""

project = 'PAV'
copyright = '2025, Peter Audano'
author = 'Peter Audano'

extensions = [
    'sphinx.ext.doctest',
    # 'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []

# AutoAPI
autoapi_dirs = [
    '../../src/pavcall',
]

autoapi_options = [
    'members',
    'undoc-members',
    # 'private-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]

autoapi_keep_files = False

autoapi_member_order = 'groupwise'

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Theme and build
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
