


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

begin
    win = 300
    ret_col_name = "ret_bp"
    th_vec = [(0.1:0.1:0.9); Float64.(1:99); (99.1:0.1:99.9)]
end

begin
    println("--------------------------------------------\n")
    ft_gen_map = Dict(
        "ofi" => ft_gen_ofi,
        "aq" => ft_gen_aq,
        "tor" => ft_gen_tor,
        "tv" => ft_gen_tv,
        "index_dist" => ft_gen_index_dist,
        "mark_dist" => ft_gen_mark_dist,
        "liquidation" => ft_gen_liqu,
        "vwap" => ft_gen_vwap,
    )
    ft_names = collect(keys(ft_gen_map))
    regime_ft_gen_map = Dict("ttc" => (ft_gen_ttc, [0.4, 0.8]), "doi" => (ft_gen_doi, [0.1, 0.9]))

    df_evv_train, df_evv_test = get_evv_df_fast2(df_train, df_test, ft_gen_map, df_train.ret_bp, th_vec)
    
    rg_info_map = get_rg_info_map(regime_ft_gen_map, df_train, df_test, df_evv_train)
    df_evv_train[!, "regime_weight_sum"] = calc_regime_based_er_vec(rg_info_map, df_evv_train, "mask_train")
    df_evv_test[!, "regime_weight_sum"] = calc_regime_based_er_vec(rg_info_map, df_evv_test, "mask_test")
end

# for th in [0.025, 0.05, 0.1, 0.5, 1.0, 5.0]
#     summary_evv_result(df_evv_train, df_evv_test, ft_names, th)
# end


th = 0.1
th = 0.33333

begin
    println("Train")
    sl_train, ss_train = get_signals(df_evv_train[!, "equal_weight_sum"], th)
    plt_train_eq, tr_res_vec_train_eq = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=false)
    println("----------")
    sl_train, ss_train = get_signals(df_evv_train[!, "lin_reg_weight_sum"], th)
    plt_train_lr, tr_res_vec_train_lr = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=false)
    println("----------")
    sl_train, ss_train = get_signals(df_evv_train[!, "regime_weight_sum"], th)
    plt_train_rg, tr_res_vec_train_rg = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_display=false)
    final_plt_train = plot(plt_train_eq, plt_train_lr, plt_train_rg, layout=(3, 1), plot_title="Train - th: $th", size=(900, 900))
end

begin
    println("Test")
    sl_test, ss_test = get_signals(df_evv_test[!, "equal_weight_sum"], th)
    plt_test_eq, tr_res_vec_test_eq = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=false)    
    println("----------")
    sl_test, ss_test = get_signals(df_evv_test[!, "lin_reg_weight_sum"], th)
    plt_test_lr, tr_res_vec_test_lr = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=false)  
    println("----------")
    sl_test, ss_test = get_signals(df_evv_test[!, "regime_weight_sum"], th)
    plt_test_rg, tr_res_vec_test_rg = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_display=false)
    final_plt_test = plot(plt_test_eq, plt_test_lr, plt_test_rg, layout=(3, 1), plot_title="Test - th: $th", size=(900, 1200))
end




begin
    er_name = "equal_weight_sum"
    plt_norm_ret_eq = view_norm_ret_path_by_df_evv(th, df_evv_train, df_evv_test, er_name=er_name)
    tr_res_vec_train_eq, tr_res_vec_test_eq = get_tr_res_vecs_by_dfs(th, df_train, df_evv_train, df_test, df_evv_test, er_name)
    plt_total_train_eq = get_plt_total_by_tr_res_vec(tr_res_vec_train_eq, "Train", df_evv_train[!, er_name], df_train[!, "WAP_Lag_0ms"], win)
    plt_total_test_eq = get_plt_total_by_tr_res_vec(tr_res_vec_test_eq, "Test", df_evv_test[!, er_name], df_test[!, "WAP_Lag_0ms"], win)
    plt_eq = plot(plt_total_train_eq, plt_total_test_eq, layout=(2, 1), plot_title="Equal Weight", size=(600, 900))
end

begin
    er_name = "lin_reg_weight_sum"
    plt_norm_ret_lr = view_norm_ret_path_by_df_evv(th, df_evv_train, df_evv_test, er_name=er_name)
    tr_res_vec_train_lr, tr_res_vec_test_lr = get_tr_res_vecs_by_dfs(th, df_train, df_evv_train, df_test, df_evv_test, er_name)
    plt_total_train_lr = get_plt_total_by_tr_res_vec(tr_res_vec_train_lr, "Train", df_evv_train[!, er_name], df_train[!, "WAP_Lag_0ms"], win)
    plt_total_test_lr = get_plt_total_by_tr_res_vec(tr_res_vec_test_lr, "Test", df_evv_test[!, er_name], df_test[!, "WAP_Lag_0ms"], win)
    plt_lr = plot(plt_total_train_lr, plt_total_test_lr, layout=(2, 1), plot_title="Lin Reg Weight", size=(600, 900))
end

begin
    er_name = "regime_weight_sum"
    plt_norm_ret_rg = view_norm_ret_path_by_df_evv(th, df_evv_train, df_evv_test, er_name=er_name)
    tr_res_vec_train_rg, tr_res_vec_test_rg = get_tr_res_vecs_by_dfs(th, df_train, df_evv_train, df_test, df_evv_test, er_name)
    plt_total_train_rg = get_plt_total_by_tr_res_vec(tr_res_vec_train_rg, "Train", df_evv_train[!, er_name], df_train[!, "WAP_Lag_0ms"], win)
    plt_total_test_rg = get_plt_total_by_tr_res_vec(tr_res_vec_test_rg, "Test", df_evv_test[!, er_name], df_test[!, "WAP_Lag_0ms"], win)
    plt_rg = plot(plt_total_train_rg, plt_total_test_rg, layout=(2, 1), plot_title="Regime Weight", size=(600, 900))
end


plt_3_norm_ret_bp = plot(plt_norm_ret_eq, plt_norm_ret_lr, plt_norm_ret_rg, layout=(1, 3), size=(900, 600))


plt_3_at_time = plot(plt_eq, plt_lr, plt_rg, layout=(1, 3), size=(1800, 900))
# display(plt_3_at_time)
savefig(plt_3_at_time, "./fig/plt_3_at_time.png")


si_train = [i for i in 1:500:size(df_evv_train, 1)]
scatter(df_evv_train[si_train, "regime_weight_sum"], df_train[si_train, "ret_bp"], title="Train Regime", markersize=2)

si_test = [i for i in 1:300:size(df_evv_test, 1)]
scatter(df_evv_test[si_test, "regime_weight_sum"], df_test[si_test, "ret_bp"], title="Test Regime", markersize=2)



begin
    position_period_train_eq = [(tr[2] - tr[1]) / 60.0 for tr in tr_res_vec_train_eq]
    position_period_test_eq = [(tr[2] - tr[1]) / 60.0 for tr in tr_res_vec_test_eq]

    position_period_train_lr = [(tr[2] - tr[1]) / 60.0 for tr in tr_res_vec_train_lr]
    position_period_test_lr = [(tr[2] - tr[1]) / 60.0 for tr in tr_res_vec_test_lr]

    position_period_train_rg = [(tr[2] - tr[1]) / 60.0 for tr in tr_res_vec_train_rg]
    position_period_test_rg = [(tr[2] - tr[1]) / 60.0 for tr in tr_res_vec_test_rg]

    # 6개 히스토그램을 직선 그래프로 합쳐서 그리기
    bins = 300
    ppp = plot()
    stephist!(ppp, position_period_train_eq, bins=bins, label="Train EQ", linewidth=2, alpha=0.7, color=:red)
    stephist!(ppp, position_period_test_eq, bins=bins, label="Test EQ", linewidth=2, alpha=0.7, color=:orange)
    stephist!(ppp, position_period_train_lr, bins=bins, label="Train LR", linewidth=2, alpha=0.7, color=:green)
    stephist!(ppp, position_period_test_lr, bins=bins, label="Test LR", linewidth=2, alpha=0.7, color=:blue)
    stephist!(ppp, position_period_train_rg, bins=bins, label="Train RG", linewidth=2, alpha=0.7, color=:black)
    stephist!(ppp, position_period_test_rg, bins=bins, label="Test RG", linewidth=2, alpha=0.7, color=:gray)
    title!(ppp, "Position Period Distribution Comparison")
    xlabel!(ppp, "Position Period (time units)")
    ylabel!(ppp, "Frequency")
    xlims!(ppp, (-5, 70))
    display(ppp)
end







