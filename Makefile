.PHONY: image init env testdata test_backtester test_scraper scrape aggregate backup bench

image:
	docker build -t data_scraper -f ./docker/data_scraper/Dockerfile .

ops:
	docker-compose -f ./docker/docker-compose.yml up -d

stop:
	docker-compose -f ./docker/docker-compose.yml down

init:
	pipenv --three && pipenv install

env:
	pipenv shell

testdata:
	pipenv run python backtester/test/create_test_data.py

test_backtester:
	pipenv run python -m pytest backtester/test

test_scraper:
	pipenv run python -m unittest discover -s data_scraper

scrape:
ifdef scraper
	pipenv run python -m data_scraper -s $(scraper) -v
else
	pipenv run python -m data_scraper -v
endif

aggregate:
	pipenv run python -m data_scraper -a

backup:
	pipenv run python -m data_scraper -b
	
bench:
	pipenv run python backtester/test/run_benchmark.py
