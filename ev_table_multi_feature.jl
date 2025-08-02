

include("const.jl")
include("utils.jl")
include("feature_generator/feature_generator.jl")

begin
    date_list_train = [
        # [string(di) for di in 20230313:20230331]; 
        [string(di) for di in 20230401:20230430];
        [string(di) for di in 20230501:20230531]
        ]
    df_list = [get_df_oneday_full(tardis_dir, s7_dir, symbol, date_one_day, features) for date_one_day in date_list_train]
    df_train = vcat(df_list...)
    set_ret_bp(df_train, ret_interval)
end

begin
    date_list_test = [
        [string(di) for di in 20230525:20230531];
        [string(di) for di in 20230601:20230630]
        ]

    df_list2 = [get_df_oneday_full(tardis_dir, s7_dir, symbol, date_one_day, features) for date_one_day in date_list_test]
    df_test = vcat(df_list2...)
    set_ret_bp(df_test, ret_interval)
end

################## from map

ret_col_name = "ret_bp"
th_vec = [(0.1:0.1:0.9); Float64.(1:99); (99.1:0.1:99.9)]

begin
    println("--------------------------------------------\n")
    ft_gen_map = Dict(
        "ofi" => ft_gen_ofi,
        "aq" => ft_gen_aq,
        "tor" => ft_gen_tor,
        "tv" => ft_gen_tv,
        "index_dist" => ft_gen_index_dist,
        "mark_dist" => ft_gen_mark_dist,
        # "trade_cnt_ratio" => ft_gen_trcr,
        # "trade_vol" => ft_gen_trv,
        "liquidation" => ft_gen_liqu,
        "vwap" => ft_gen_vwap,
    )
    ft_names = collect(keys(ft_gen_map))

    # df_evv_train, df_evv_test = get_evv_df(ft_gen_map, ret_col_name, th_vec)
    df_evv_train, df_evv_test = get_evv_df_fast2(df_train, df_test, ft_gen_map, df_train.ret_bp, th_vec)
    for th in [0.025, 0.05, 0.1, 0.5, 1.0, 5.0]
        summary_evv_result(df_evv_train, df_evv_test, ft_names, th)
    end
end


win = 300
th = 0.025
th = 0.05
th = 0.1
th = 0.5
th = 1.0

plt_norm_ret = view_norm_ret_path_by_df_evv(th, df_evv_train, df_evv_test, df_train, df_test)
tr_res_vec_train, tr_res_vec_test = get_tr_res_vecs_by_dfs(th, df_train, df_evv_train, df_test, df_evv_test)

plt_total_train = get_plt_total_by_tr_res_vec(tr_res_vec_train, "Train", df_evv_train[!, "equal_weight_sum"], df_train[!, "WAP_Lag_0ms"], win)
plt_total_test = get_plt_total_by_tr_res_vec(tr_res_vec_test, "Test", df_evv_test[!, "equal_weight_sum"], df_test[!, "WAP_Lag_0ms"], win)


begin
    sl_train, ss_train = get_signals(df_evv_train[!, "equal_weight_sum"], th)
    plt_train, tr_res_vec_train = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=true)
end

begin
    sl_test, ss_test = get_signals(df_evv_test[!, "equal_weight_sum"], th)
    plt_test, tr_res_vec_test = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=true)
end



coef_lr_train = get_lr_coef(df_evv_train, df_train.ret_bp)
yhat_lr_train = calc_er_vec_by_coef(df_evv_train, coef_lr_train)
begin
    sl_lr, ss_lr = get_signals(yhat_lr_train, th)
    plt, tr_res_vec_train = backtest_sica_2(sl_lr, ss_lr, df_train[!, :WAP_Lag_200ms], df_train[!, :WAP_Lag_0ms], df_train[!, :timestamp], is_display=false)
    title!(plt, "Linear Reg - Train")
    display(plt)
end


# coef_lr_test = get_lr_coef(df_evv_test, df_test.ret_bp)
yhat_lr_test = calc_er_vec_by_coef(df_evv_test, coef_lr_train)
begin
    sl_lr, ss_lr = get_signals(yhat_lr_test, th)
    plt, tr_res_vec_test = backtest_sica_2(sl_lr, ss_lr, df_test[!, :WAP_Lag_200ms], df_test[!, :WAP_Lag_0ms], df_test[!, :timestamp], is_display=false)
    title!(plt, "Linear Reg - Test")
    display(plt)
end


begin
    wap = df_train.WAP_Lag_0ms
    avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec_train, wap)
    plt_test = plot(avg_norm_paths, label="avg")
    plot!(plt_test, median_norm_paths, label="median")
    title!(plt_test, "Train")
end

begin
    wap = df_test.WAP_Lag_0ms
    avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec_test, wap)
    plt_test = plot(avg_norm_paths, label="avg")
    plot!(plt_test, median_norm_paths, label="median")
    title!(plt_test, "Test")
end

plt_total_train = get_plt_total_by_tr_res_vec(tr_res_vec_train, "Train", yhat_lr_train, df_train[!, "WAP_Lag_0ms"], win)
plt_total_test = get_plt_total_by_tr_res_vec(tr_res_vec_test, "Test", yhat_lr_test, df_test[!, "WAP_Lag_0ms"], win)



# 내일 할 일
# 1. 레짐 나눠서 해보자 
# 2. 심볼별로 확장 


# 1. feature for regime divide (trade count, diff of open interest)























weights = Dict(cf[1] => cf[2] for cf in coef_lr_train.coefs)
df_evv_train[!, "weighted_sum"] = calc_weighted_sum(df_evv_train, weights, coef_lr_train.intercept)
df_evv_test[!, "weighted_sum"] = calc_weighted_sum(df_evv_test, weights, coef_lr_train.intercept)


# 예시: 레짐별 가중치 적용
# 레짐은 예를 들어 변동성이나 거래량 등에 따라 정의할 수 있음
# 여기서는 임의로 2개의 레짐을 가정
regime_vec_train = rand(1:2, size(df_evv_train, 1))  # 실제로는 적절한 레짐 구분 로직 필요
regime_vec_test = rand(1:2, size(df_evv_test, 1))

regime_weights = Dict(
    1 => Dict("ofi" => 0.8, "aq" => 0.2),  # 저변동성 레짐
    2 => Dict("ofi" => 0.3, "aq" => 0.7)   # 고변동성 레짐
)

df_evv_train[!, "regime_weighted_sum"] = calc_regime_weighted_sum(df_evv_train, regime_vec_train, regime_weights, ft_names)
df_evv_test[!, "regime_weighted_sum"] = calc_regime_weighted_sum(df_evv_test, regime_vec_test, regime_weights, ft_names)

summary_evv_result(df_evv_train, df_evv_test, ft_names, 1.0)




# th = 5.0
# er_bps_train, er_bps_test = [], []
# for ft_name in ft_names
#     sl_train, ss_train = get_signals(df_evv_train[!, ft_name], th)
#     plt_train, tr_res_vec_train = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_print=false)
#     er_bp_train = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_train])
#     push!(er_bps_train, er_bp_train)

#     sl_test, ss_test = get_signals(df_evv_test[!, ft_name], th)
#     plt_test, tr_res_vec_test = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_print=false)
#     er_bp_test = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_test])
#     push!(er_bps_test, er_bp_test)
# end

# sl_train, ss_train = get_signals(df_evv_train.equal_weight_sum, th)
# plt_train, tr_res_vec_train = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_print=false)
# er_bp_train = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_train])
# push!(er_bps_train, er_bp_train)

# sl_test, ss_test = get_signals(df_evv_test.equal_weight_sum, th)
# plt_test, tr_res_vec_test = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_print=false)
# er_bp_test = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_test])
# push!(er_bps_test, er_bp_test)


# train_bps_str = join(string.(round.(er_bps_train[1 : end-1], digits=2)), ", ")
# test_bps_str = join(string.(round.(er_bps_test[1 : end-1], digits=2)), ", ")
# println("th: $th  $(join(ft_names, "  "))")
# println("Train : [$(train_bps_str)] -> $(round(er_bps_train[end], digits=2))")
# println("Test  : [$(test_bps_str)] -> $(round(er_bps_test[end], digits=2))")













th = 5.0
is_display = true



sl, ss = get_signals(df_evv_train[!, "ofi"], th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=is_display)

sl, ss = get_signals(df_evv_train[!, "aq"], th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=is_display)

sl, ss = get_signals(df_evv_train.equal_weight_sum, th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=is_display)

sl, ss = get_signals(df_evv_train.weighted_sum, th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=is_display)

sl, ss = get_signals(df_evv_train.regime_weighted_sum, th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=is_display)


simple_view_feature_power(df_evv_train.equal_weight_sum, df_train[!, ret_col_name])






sl, ss = get_signals(df_evv_test[!, "ofi"], th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=is_display)

sl, ss = get_signals(df_evv_test[!, "aq"], th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=is_display)

sl, ss = get_signals(df_evv_test.equal_weight_sum, th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=is_display)

sl, ss = get_signals(df_evv_test.weighted_sum, th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=is_display)

sl, ss = get_signals(df_evv_test.regime_weighted_sum, th)
plt, tr_res_vec = backtest_sica_2(sl, ss, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=is_display)









