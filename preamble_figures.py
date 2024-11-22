import seaborn as sns
import matplotlib.pyplot as plt
TITLE_SIZE = 18
LEGEND_SIZE = 15
TICK_SIZE = 15
AXIS_TITLE = 18
AXIS_LABEL = 15
FONT_SIZE = 15


rc_parameters = {
    "font.size": FONT_SIZE,
    "axes.titlesize": AXIS_TITLE,
    "axes.labelsize": AXIS_LABEL,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": TITLE_SIZE,
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,  # use inline math for ticks
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "pgf.preamble": r'\usepackage{amsmath}\usepackage{mathtools} \usepackage{amssymb}'
}
plt.rcParams.update(rc_parameters)
plt.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",              
        r"\usepackage{mathtools}",
        r"\usepackage{amssymb}"
])

sns.set_style("darkgrid", rc=rc_parameters)