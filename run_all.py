from plotter_algorithm_demo import PlotterAlgorithm
from plotter_ui_demo import PlotterUI


def run_all():
    algorithm = PlotterAlgorithm()
    ui = PlotterUI(algorithm)
    ui.start()


if __name__ == "__main__":
    run_all()