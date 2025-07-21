

# um_path = raw"F:\Public\Binance_Vision\futures\um"
# save_dir = raw"F:\Public\Binance_Vision\futures\processed"

# symbol = "BTCUSDT"
# interval_sec = 5



dtype = "premiumIndexKlines"
file_list_pi = get_file_list(um_path, symbol, dtype)
col_names_pi = names(read_zip_to_df(file_list_pi[end]))

dtype = "aggTrades"
file_list_aggTr = get_file_list(um_path, symbol, dtype)
col_names_aggTr = names(read_zip_to_df(file_list_aggTr[end]))

dtype = "fundingRate"
file_list_fdr = get_file_list(um_path, symbol, dtype)
col_names_fdr = names(read_zip_to_df(file_list_fdr[end]))











