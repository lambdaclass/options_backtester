from python import Python

fn main() raises:
    Python.add_to_path("./")
    var backtester = Python.import_module("backtester")
    var test = Python.import_module("backtester.test")
    var sample_stocks_datahandler = test.sample_stocks_datahandler()
    var sample_options_datahandler = test.sample_options_datahandler()
    var sample_options_strategy = test.backtester.sample_options_strategy(backtester.Direction.BUY, sample_options_datahandler.schema)


    ## This code below crashes for me for some reason, even though it looks the same as
    ## running test.backtester.run_backtest(...)

    # var allocation = Python.dict()

    # allocation["stocks"] = 0.50
    # allocation["options"] = 0.50
    # allocation["cash"] = 0

    # var bt = backtester.Backtest(allocation, [])
    # bt.stocks_data = sample_stocks_datahandler
    # bt.options_strategy = sample_options_strategy
    # bt.options_data = sample_options_datahandler
    # bt.stocks = test.sample_stock_portfolio()

    # bt.run(rebalance_freq=1)

    test.backtester.run_backtest(sample_stocks_datahandler,
                      sample_options_datahandler,
                      sample_options_strategy,
                      stocks=test.sample_stock_portfolio())
