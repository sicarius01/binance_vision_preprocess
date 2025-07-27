

include("utils.jl")


tardis_dir = raw"F:\Public\Tardis_2023\datasets"
s7_dir = raw"D:\Sunday7\OnlineData"
symbol = "BTCUSDT"


date_list = [[string(di) for di in 20230313:20230331]; [string(di) for di in 20230401:20230430]; [string(di) for di in 20230501:20230531]]
df_list = [get_df_oneday_with_deri_liqu(tardis_dir, s7_dir, symbol, date_one_day) for date_one_day in date_list]
df = vcat(df_list...)













# histogram(df.open_interest)
# histogram(df.ret_bp, xlims=(-30, 30))

df[!, "wap-index_price"] .= 10_000 .* (df.index_price .- df.WAP_Lag_0ms) ./ df.WAP_Lag_0ms
histogram(df[!, "wap-index_price"])

df[!, "wap-mark_price"] .= 10_000 .* (df.mark_price .- df.WAP_Lag_0ms) ./ df.WAP_Lag_0ms
histogram(df[!, "wap-mark_price"])

simple_view_feature_power(df[!, "wap-index_price"], df.ret_bp)
simple_view_feature_power(df[!, "wap-mark_price"], df.ret_bp)

si = [i for i in 1:600:size(df, 1)]
plot(df[si, "wap-index_price"], df[si, "ret_bp"], seriestype = :scatter, markersize=2)
plot(df[si, "wap-mark_price"], df[si, "ret_bp"], seriestype = :scatter, markersize=2)

f1 = df[!, "wap-index_price"]
f2 = df[!, "wap-mark_price"]

pct = 1.0
sl, ss = get_signals(f1, pct)

backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp)


pct = 1.0
sl, ss = get_signals(f2, pct)

backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp)



f1_norm = norm_by_before_n_days(f1, 7, 1)
f2_norm = norm_by_before_n_days(f2, 7, 1)
histogram(f1_norm, xlims=(-10, 10))
histogram(f2_norm, xlims=(-10, 10))

simple_view_feature_power(f1_norm, df.ret_bp)
simple_view_feature_power(f2_norm, df.ret_bp)

pct = 0.1
sl, ss = get_signals(f1_norm, pct)
backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp)

sl, ss = get_signals(f2_norm, pct)
backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp)




f1[isnan.(f1)] .= 0.0
f2[isnan.(f2)] .= 0.0

f1n2 = ema_norm(f1, 15, 1800)
f2n2 = ema_norm(f2, 15, 1800)

f1n2 = ema_norm(f1, 60, 7200)
f2n2 = ema_norm(f2, 60, 7200)

f1n2 = ema_norm(f1, 15, 900)
f2n2 = ema_norm(f2, 15, 900)
histogram(f1n2)
histogram(f2n2)

plot(f1n2[si], df.ret_bp[si], seriestype = :scatter, title="index-wap_bp")
plot(f2n2[si], df.ret_bp[si], seriestype = :scatter, title="mark-wap_bp")

simple_view_feature_power(f1n2, df.ret_bp)
simple_view_feature_power(f2n2, df.ret_bp)

pct = 0.5
sl, ss = get_signals(f1n2, pct, th_pct2=0.0)
backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol)

sl, ss = get_signals(f2n2, pct, th_pct2=0.0)
backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol)




