import os
import backtester as bt
from backtester.utils import get_data_dir

data_dir = get_data_dir()
spx_test_data = os.path.join(data_dir, "SPX_2008-2018.csv")

portfolio = bt.run(spx_test_data)
report = portfolio.create_report()

print("Running Benchmark strategy on SPX data for 2008-2018")
print(report.tail(30))
