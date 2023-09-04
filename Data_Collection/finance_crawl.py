import FinanceDataReader as fdr

kospi_data = fdr.StockListing("KOSPI")
kosdaq_data = fdr.StockListing("KOSDAQ")

stock_mapping = {}
for _, row in kospi_data.iterrows():
    stock_mapping[row["Name"]] = row["Symbol"]
for _, row in kosdaq_data.iterrows():
    stock_mapping[row["Name"]] = row["Symbol"]

print(stock_mapping)