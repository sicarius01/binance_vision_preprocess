

include("utils.jl")
include("const.jl")


tardis_dir = raw"F:\Public\Tardis_2023\datasets"
s7_dir = raw"D:\Sunday7\OnlineData"

symbol = "DOGEUSDT"

date_list = [
    [string(di) for di in 20230313:20230331]; 
    [string(di) for di in 20230401:20230430];
    [string(di) for di in 20230501:20230431];
    [string(di) for di in 20230601:20230430]
    ]
df_list = [get_df_oneday_with_deri_liqu(tardis_dir, s7_dir, symbol, date_one_day) for date_one_day in date_list]
df = vcat(df_list...)
set_ret_bp(df, 15)






df[!, "wap-index_price"] .= 10_000 .* (df.index_price .- df.WAP_Lag_0ms) ./ df.WAP_Lag_0ms
df[!, "wap-mark_price"] .= 10_000 .* (df.mark_price .- df.WAP_Lag_0ms) ./ df.WAP_Lag_0ms
f1 = df[!, "wap-index_price"]
f2 = df[!, "wap-mark_price"]
si = [i for i in 1:600:size(df, 1)]


f1[isnan.(f1)] .= 0.0
f2[isnan.(f2)] .= 0.0

f1n2 = ema_norm(f1, 15, 1800)
f2n2 = ema_norm(f2, 15, 1800)

f1n2 = ema_norm(f1, 60, 7200)
f2n2 = ema_norm(f2, 60, 7200)

f1n2 = ema_norm(f1, 15, 900)
f2n2 = ema_norm(f2, 15, 900)


pct = 0.5
sl, ss = get_signals(f1n2, pct, th_pct2=0.0)
backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, 15)
plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, is_display=true)

sl, ss = get_signals(f2n2, pct, th_pct2=0.0)
backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, 15)
plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, is_display=true)







wap = df.WAP_Lag_0ms
avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec, wap)

begin
    plot(avg_norm_paths, label="avg")
    plot!(median_norm_paths, label="median")
end


avg_pr_paths, median_pr_paths, remainer = get_full_ret_bp_path(tr_res_vec, wap)

begin
    plot(avg_pr_paths, label="avg")
    # plot!(median_pr_paths, label="median")
    plot!(twinx(), remainer, label="remain", color=:green)
end














histogram(df.open_interest)
histogram(df.ret_bp, xlims=(-30, 30))

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
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)


pct = 1.0
sl, ss = get_signals(f2, pct)

backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)



f1_norm = norm_by_before_n_days(f1, 7, 1)
f2_norm = norm_by_before_n_days(f2, 7, 1)
histogram(f1_norm, xlims=(-10, 10))
histogram(f2_norm, xlims=(-10, 10))

simple_view_feature_power(f1_norm, df.ret_bp)
simple_view_feature_power(f2_norm, df.ret_bp)

pct = 0.1
sl, ss = get_signals(f1_norm, pct)
backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)

sl, ss = get_signals(f2_norm, pct)
backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15)
backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)




# f1[isnan.(f1)] .= 0.0
# f2[isnan.(f2)] .= 0.0

# f1n2 = ema_norm(f1, 15, 1800)
# f2n2 = ema_norm(f2, 15, 1800)

# f1n2 = ema_norm(f1, 60, 7200)
# f2n2 = ema_norm(f2, 60, 7200)

# f1n2 = ema_norm(f1, 15, 900)
# f2n2 = ema_norm(f2, 15, 900)
# histogram(f1n2)
# histogram(f2n2)

# plot(f1n2[si], df.ret_bp[si], seriestype = :scatter, title="index-wap_bp")
# plot(f2n2[si], df.ret_bp[si], seriestype = :scatter, title="mark-wap_bp")

# simple_view_feature_power(f1n2, df.ret_bp)
# simple_view_feature_power(f2n2, df.ret_bp)

# pct = 0.5
# sl, ss = get_signals(f1n2, pct, th_pct2=0.0)
# backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, 15)
# plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, is_display=true)

# sl, ss = get_signals(f2n2, pct, th_pct2=0.0)
# backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, 15)
# backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, is_display=true)

















# df[!, "index_distance_bp"] = 10_000 .* (df.WAP_Lag_0ms .- df.index_price) ./ df.WAP_Lag_0ms
# df[!, "mark_distance_bp"] = 10_000 .* (df.WAP_Lag_0ms .- df.mark_price) ./ df.WAP_Lag_0ms

# si = [i for i in 1:900:size(df, 1)]
# plot(df[si, "index_distance_bp"], df[si, :ret_bp], seriestype = :scatter, markersize=2)
# plot(df[si, "mark_distance_bp"], df[si, :ret_bp], seriestype = :scatter, markersize=2)


# ss, sl = get_signals(df[!, "index_distance_bp"], 0.5)
# plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)

# ss, sl = get_signals(df[!, "mark_distance_bp"], 1)
# plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)


# wap = df.WAP_Lag_0ms
# avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec, wap)

# begin
#     plot(avg_norm_paths, label="avg")
#     plot!(median_norm_paths, label="median")
# end


# avg_pr_paths, median_pr_paths, remainer = get_full_ret_bp_path(tr_res_vec, wap)

# begin
#     plot(avg_pr_paths, label="avg")
#     # plot!(median_pr_paths, label="median")
#     plot!(twinx(), remainer, label="remain", color=:green)
# end














# #################################################################
# using DuckDB

# function read_duckdb_to_df(db_path::String, query::String)
#     con = DBInterface.connect(DuckDB.DB, db_path)
#     try
#         df = DataFrame(DBInterface.execute(con, query))
#     finally
#         DBInterface.close!(con)
#     end
#     return df
# end


# ddb_path = raw"C:\Users\haeso\Documents\project\binance_vision_preprocess\data\tardis_derivative_liquidations"

# db_files = [dbp for dbp in readdir(ddb_path) if occursin("DOGEUSDT", dbp)]

# dfs = [read_duckdb_to_df(db_path, "SHOW TABLES") for db_path in db_files[2:end]]
# df_ddb = vcat(dfs...)

# Int(df_ddb[1, 1])
# Int(df_ddb[2, 1])
# Int(df_ddb[3, 1])
# Int(df_ddb[4, 1])
# Int(df_ddb[end, 1])

# for cn in names(df_ddb)
#     println(cn)
# end


# df = read_duckdb_to_df(db_files[1], "SHOW TABLES")
# set_ret_bp(df, 15)

# unix2datetime(Int(df[1, 1] ÷ 1000))
# unix2datetime(Int(df[2, 1] ÷ 1000))
# unix2datetime(Int(df[3, 1] ÷ 1000))
# unix2datetime(Int(df[4, 1] ÷ 1000))
# unix2datetime(Int(df[end, 1] ÷ 1000))

# df[!, "index_distance_bp"] = 10_000 .* (df.WAP_Lag_0ms .- df.index_distance_bp) ./ df.WAP_Lag_0ms
# df[!, "mark_distance_bp"] = 10_000 .* (df.WAP_Lag_0ms .- df.mark_distance_bp) ./ df.WAP_Lag_0ms

# si = [i for i in 1:300:size(df, 1)]
# plot(df[si, "index_distance_bp"], df[si, :ret_bp], seriestype = :scatter, markersize=1)
# plot(df[si, "mark_distance_bp"], df[si, :ret_bp], seriestype = :scatter, markersize=1)

# simple_view_feature_power(df[!, "index_distance_bp"], df.ret_bp)
# simple_view_feature_power(df[!, "mark_distance_bp"], df.ret_bp)

# sl, ss = get_signals(df[!, "index_distance_bp"], 0.5)
# plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)

# sl, ss = get_signals(df[!, "mark_distance_bp"], 0.5)
# plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)



# wap = df.WAP_Lag_0ms
# avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec, wap)

# begin
#     plot(avg_norm_paths, label="avg")
#     plot!(median_norm_paths, label="median")
# end


# avg_pr_paths, median_pr_paths, remainer = get_full_ret_bp_path(tr_res_vec, wap)

# begin
#     plot(avg_pr_paths, label="avg")
#     # plot!(median_pr_paths, label="median")
#     plot!(twinx(), remainer, label="remain", color=:green)
# end