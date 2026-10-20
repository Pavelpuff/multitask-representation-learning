.PHONY: download-data

download-data:
	curl -LO https://github.com/marcin119a/data/raw/refs/heads/main/data_gsn.zip
	unzip -o data_gsn.zip
	rm data_gsn.zip
	@echo "Dataset successfully downloaded and extracted to data/"