

include("utils.jl")


um_path = raw"C:\Users\haeso\Documents\project\Binance_Futures_Download\binance-vision-data\futures\um"
symbol = "BTCUSDT"
dtype = "premiumIndexKlines"


file_list_pi = get_file_list(um_path, symbol, dtype)
df_pi = read_zip_to_df(file_list_pi[1])



dtype = "aggTrades"
file_list_aggTr = get_file_list(um_path, symbol, dtype)
df_aggTr = read_zip_to_df(file_list_aggTr[1])



dtype = "fundingRate"
file_list_fdr = get_file_list(um_path, symbol, dtype)
df_fdr = read_zip_to_df(file_list_fdr[1])



dts = diff(df_aggTr.transact_time)
histogram(dts[dts .> 0], xlims=(-30, 1000))
minimum(dts[dts .> 0])




# timestamp는 ms 단위
# aggTrades는 모이는대로 찍히고, premiumIndexKlines는 분 단위, fundingRate는 8시간 단위
# aggTrades만 전처리 하자. 1초봉을 만들자. ohlcv 그리고 total_id_cnt, buy_flow_id_cnt, sell_flow_id_cnt, vwap까지 만들면 될듯 



df_aggTr.price

df_aggTr




