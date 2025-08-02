


include("const.jl")
include("utils.jl")


date_list = [
    [string(di) for di in 20230313:20230331]; 
    [string(di) for di in 20230401:20230430];
    [string(di) for di in 20230501:20230531]
    ]
df_list = [get_df_oneday_full(tardis_dir, s7_dir, symbol, date_one_day, features) for date_one_day in date_list]
df = vcat(df_list...)
set_ret_bp(df, ret_interval)


of_b = df[!, "orderflow_v2_bid_place_volume_0_bp"] .- df[!, "orderflow_v2_bid_cancel_volume_0_bp"]
of_a = df[!, "orderflow_v2_ask_place_volume_0_bp"] .- df[!, "orderflow_v2_ask_cancel_volume_0_bp"]
of_b_norm = norm_by_before_n_days(of_b, 7, 1)
of_a_norm = norm_by_before_n_days(of_a, 7, 1)
ofi = of_b_norm .- of_a_norm


df[!, "wap-index_price"] .= 10_000 .* (df.index_price .- df.WAP_Lag_0ms) ./ df.WAP_Lag_0ms
df[!, "wap-mark_price"] .= 10_000 .* (df.mark_price .- df.WAP_Lag_0ms) ./ df.WAP_Lag_0ms
f1 = df[!, "wap-index_price"]
f2 = df[!, "wap-mark_price"]
si = [i for i in 1:600:size(df, 1)]

f1[isnan.(f1)] .= 0.0
f2[isnan.(f2)] .= 0.0
f1n2 = ema_norm(f1, 15, 1800)
f2n2 = ema_norm(f2, 15, 1800)


sl_entry, ss_entry = get_signals(ofi, 0.5)
# ss_exit, sl_exit = get_signals(f1n2, 0.5)
ss_exit, sl_exit = get_signals(f1, 0.5)
plt, tr_res_vec = backtest_sica_3(sl_entry, sl_exit, ss_entry, ss_exit, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, symbol=symbol, is_display=true)


wap = df.WAP_Lag_0ms
avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec, wap)
begin
    plot(avg_norm_paths, label="avg")
    plot!(median_norm_paths, label="median")
    title!("Complex Backtest Norm Ret(bp) Path")
end


# avg_pr_paths, median_pr_paths, remainer = get_full_ret_bp_path(tr_res_vec, wap)
# begin
#     plot(avg_pr_paths, label="avg")
#     # plot!(median_pr_paths, label="median")
#     plot!(twinx(), remainer, label="remain", color=:green)
#     title!("Complex Backtest Full Ret(bp) Path")
# end

# ltr = [tr for tr in tr_res_vec if tr[3] > 0]
# str = [tr for tr in tr_res_vec if tr[3] < 0]

# begin
#     avg_norm_paths_f, median_norm_paths_f = get_norm_feature_path(ltr, f1n2)
#     avg_norm_paths, median_norm_paths = get_norm_feature_path(ltr, wap)
#     plot(avg_norm_paths_f, label="feature")
#     plot!(avg_norm_paths, label="ret")
#     title!("Long")
# end

# begin
#     avg_norm_paths_f, median_norm_paths_f = get_norm_feature_path(str, f1n2)
#     avg_norm_paths, median_norm_paths = get_norm_feature_path(str, wap)
#     plot(avg_norm_paths_f, label="feature")
#     plot!(avg_norm_paths, label="ret")
#     title!("Short")
# end



win = 500
idx_at_list = [tr[1] for tr in tr_res_vec if tr[3] > 0]
idx_at_list = [tr[2] for tr in tr_res_vec if tr[3] > 0]
avg_fvs, avg_waps, median_fvs, median_waps = get_at_time_profile(f1n2, wap, win, idx_at_list)
avg_fvs_ofi, avg_waps, median_fvs, median_waps = get_at_time_profile(ofi, wap, win, idx_at_list)

begin
    plot(-win:win, avg_fvs, label="f1n2", color=:green)
    plot!(twinx(), -win:win, avg_fvs_ofi, label="ofi")
    plot!([0], [avg_fvs[win+1]], seriestype = :scatter)
    title!("Long - fv [ofi, f1n2]")
end
begin
    plot(-win:win, avg_waps, label="wap", title="Long - wap [ofi, f1n2]")
    plot!([0], [avg_waps[win+1]], seriestype = :scatter)
end




win = 500
idx_at_list = [tr[1] for tr in tr_res_vec if tr[3] < 0]
idx_at_list = [tr[2] for tr in tr_res_vec if tr[3] < 0]
avg_fvs, avg_waps, median_fvs, median_waps = get_at_time_profile(f1n2, wap, win, idx_at_list)
avg_fvs_ofi, avg_waps, median_fvs, median_waps = get_at_time_profile(ofi, wap, win, idx_at_list)

begin
    plot(-win:win, avg_fvs, label="f1n2", color=:green)
    plot!(twinx(), -win:win, avg_fvs_ofi, label="ofi")
    plot!([0], [avg_fvs[win+1]], seriestype = :scatter)
    title!("Short - fv [ofi, f1n2]")
end
begin
    plot(-win:win, avg_waps, label="wap", title="Short - wap [ofi, f1n2]")
    plot!([0], [avg_waps[win+1]], seriestype = :scatter)
end



