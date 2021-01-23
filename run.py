import warnings
warnings.simplefilter('ignore')

import strategy
from control import Controller


agent = strategy.EdgeDetectionAgent(10)
# agent = strategy.SiameseNetAgent(10)
controller = Controller(agent, data_dir=None)
controller.run()
