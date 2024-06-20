import argparse
import coloredlogs, logging
import traceback

import ft_plotter

def get_args():
	# script_dir = os.path.dirname(os.path.abspath(__file__))

	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('historics_directory', nargs=1)
	return parser.parse_args()

def main():
	# get args and initialize logging
	coloredlogs.install()
	args = get_args()
	if args.verbose:
		coloredlogs.set_level(logging.DEBUG)
	try:
		plotter = ft_plotter.Ft_plotter(args.historics_directory[0])
		logging.debug("plotter created")

		plotter.read_values()
		logging.debug("values read")

		plotter.plot_diagrams()
		logging.info("Diagram plotted")
	except Exception as e:
		logging.error(f"ft_plotter: {e}")
		logging.error(traceback.format_exc())

main()