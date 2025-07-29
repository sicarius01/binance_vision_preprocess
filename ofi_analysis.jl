


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
simple_view_feature_power(of_b_norm, df.ret_bp)
simple_view_feature_power(of_a_norm, df.ret_bp)

ofi = of_b_norm .- of_a_norm
histogram(ofi, xlims=(-7, 7))
simple_view_feature_power(ofi, df.ret_bp)

pct = 0.50
sl, ss = get_signals(ofi, pct)
plt, tr_res_vec = backtest_sica_keep(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15, is_display=true)
plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)

wap = df.WAP_Lag_0ms
avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec, wap)
begin
    plot(avg_norm_paths, label="avg")
    plot!(median_norm_paths, label="median")
end


avg_pr_paths, median_pr_paths, remainer = get_full_ret_bp_path(tr_res_vec, wap)
begin
    plot(avg_pr_paths, label="avg")
    plot!(median_pr_paths, label="median")
    plot!(twinx(), remainer, label="remain")
end


period = [tr_res[2] - tr_res[1] for tr_res in tr_res_vec]
histogram(period ./ 60)



ltr = [tr for tr in tr_res_vec if tr[3] > 0]
str = [tr for tr in tr_res_vec if tr[3] < 0]

begin
    avg_norm_paths_f, median_norm_paths_f = get_norm_feature_path(ltr, ofi)
    avg_norm_paths, median_norm_paths = get_norm_feature_path(ltr, wap)
    plot(avg_norm_paths_f, label="feature")
    plot!(avg_norm_paths, label="ret")
    title!("Long")
end

begin
    avg_norm_paths_f, median_norm_paths_f = get_norm_feature_path(str, ofi)
    avg_norm_paths, median_norm_paths = get_norm_feature_path(str, wap)
    plot(avg_norm_paths_f, label="feature")
    plot!(avg_norm_paths, label="ret")
    title!("Short")
end





win = 500
idx_at_list = [tr[1] for tr in tr_res_vec if tr[3] > 0]
avg_fvs, avg_waps, median_fvs, median_waps = get_at_time_profile(ofi, wap, win, idx_at_list)

begin
    plot(-win:win, avg_fvs, label="fv", title="Long - fv [ofi]")
    plot!([0], [avg_fvs[win+1]], seriestype = :scatter)
end
begin
    plot(-win:win, avg_waps, label="wap", title="Long - wap [ofi]")
    plot!([0], [avg_waps[win+1]], seriestype = :scatter)
end




win = 500
idx_at_list = [tr[1] for tr in tr_res_vec if tr[3] < 0]
avg_fvs, avg_waps, median_fvs, median_waps = get_at_time_profile(ofi, wap, win, idx_at_list)

begin
    plot(-win:win, avg_fvs, label="fv", title="Short - fv [ofi]")
    plot!([0], [avg_fvs[win+1]], seriestype = :scatter)
end
begin
    plot(-win:win, avg_waps, label="wap", title="Short - wap [ofi]")
    plot!([0], [avg_waps[win+1]], seriestype = :scatter)
end





