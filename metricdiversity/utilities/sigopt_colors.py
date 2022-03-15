"""
NOTE(Harvey)
This has been updated with our current brand colors (as of 6/4/19). See brand guideline:
https://drive.google.com/drive/u/1/folders/12uCg4OIddepRQd700hJYIsM8P_PJ5-s2

NOTE(Mike)
I have left the deprecated colors in as well,
in case we need them to recreate an image from earlier in our existence.
"""
import matplotlib as _mpl

sigopt_blue = '#245EAB'
sigopt_light_blue = '#0098D1'
sigopt_dark_blue = '#0B3267'
sigopt_white = '#FFFFFF'
sigopt_purple = '#772A90'
sigopt_magenta = '#A23D97'
sigopt_orange = '#F5811F'
sigopt_yellow = '#FFE739'
sigopt_olive = '#96CA4F'
sigopt_green = '#00B253'
sigopt_charcoal = '#343740'
sigopt_black = '#000000'

sigopt_colors_for_cmap_1 = [
    sigopt_blue,
    sigopt_orange,
]
cmap_sigopt = _mpl.colors.LinearSegmentedColormap.from_list(
    'brand_colors',
    sigopt_colors_for_cmap_1,
    N=200,
)
sigopt_colors_for_cmap_2 = [
    sigopt_dark_blue,
    sigopt_purple,
]
cmap_sigopt_cool = _mpl.colors.LinearSegmentedColormap.from_list(
    'brand_colors_cool',
    sigopt_colors_for_cmap_2,
    N=200,
)
sigopt_colors_for_cmap_3 = [
    sigopt_charcoal,
    sigopt_orange,
    sigopt_yellow,
]
cmap_sigopt_hot = _mpl.colors.LinearSegmentedColormap.from_list(
    'brand_colors',
    sigopt_colors_for_cmap_3,
    N=200,
)

sigopt_colors_for_full_cycle = [
    sigopt_blue,
    sigopt_purple,
    sigopt_green,
    sigopt_orange,
    sigopt_light_blue,
    sigopt_dark_blue,
    sigopt_yellow,
    sigopt_magenta,
    sigopt_olive,
    sigopt_charcoal,
]


def cmap_sigopt_manufactured(num_colors):
    colors_to_use = sigopt_colors_for_full_cycle * (1 + num_colors // len(sigopt_colors_for_full_cycle))
    new_cmap = _mpl.colors.LinearSegmentedColormap.from_list(
        'all_brand_colors_repeated',
        colors_to_use[:num_colors],
        N=num_colors,
    )
    return new_cmap

###################################

# _DEPRECATED_sigopt_aqua = '#00a499'
_DEPRECATED_sigopt_blue = '#1F407D'
_DEPRECATED_sigopt_red = '#E84557'
_DEPRECATED_sigopt_orange = '#F89B20'
_DEPRECATED_sigopt_black = '#000000'
_DEPRECATED_sigopt_aqua = '#0098DB'  # Not sure how I had two of these ...
_DEPRECATED_sigopt_gray = '#F7F7FA'
_DEPRECATED_sigopt_sky = '#9FC7DC'
_DEPRECATED_sigopt_ash = '#6F8691'
_DEPRECATED_sigopt_neon_blue = '#5AC0DE'
_DEPRECATED_nvidia_green = '#76BA02'
_DEPRECATED_sigopt_white = '#FFFFFF'

_DEPRECATED_sigopt_colors_for_cmap = [
    '#1f407d',
    '#2291cf',
    '#f89b20',
    '#fbbf15',
]
_DEPRECATED_cmap_sigopt = _mpl.colors.LinearSegmentedColormap.from_list(
    '_DEPRECATED_brand_colors',
    _DEPRECATED_sigopt_colors_for_cmap,
    N=200,
)