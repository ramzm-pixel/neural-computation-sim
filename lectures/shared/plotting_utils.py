import os

def get_figures_dir(script_file):
    """
    Given a script's __file__, return (and create) a 'figures' folder next
    to that script - so savefig() paths resolve relative to the script's
    own location, not whatever directory it happened to be run from.

    Call as: FIGURES_DIR = get_figures_dir(__file__)
    """
    script_dir = os.path.dirname(os.path.abspath(script_file))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir