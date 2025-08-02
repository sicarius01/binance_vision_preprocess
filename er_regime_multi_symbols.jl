

include("const.jl")
include("utils.jl")
include("feature_generator/feature_generator.jl")

begin
    df_train_map::Dict{String, DataFrame} = Dict()
    for sb in symbols
        date_list_train = [
            # [string(di) for di in 20230313:20230331]; 
            [string(di) for di in 20230401:20230430];
            [string(di) for di in 20230501:20230531]
            ]
        df_list = [get_df_oneday_full(tardis_dir, s7_dir, sb, date_one_day, features) for date_one_day in date_list_train]
        df_train = vcat(df_list...)
        set_ret_bp(df_train, ret_interval)
        df_train_map[sb] = df_train
    end
end

begin
    df_test_map::Dict{String, DataFrame} = Dict()
    for sb in symbols
        date_list_test = [
            [string(di) for di in 20230525:20230531];
            [string(di) for di in 20230601:20230630]
            ]

        df_list2 = [get_df_oneday_full(tardis_dir, s7_dir, sb, date_one_day, features) for date_one_day in date_list_test]
        df_test = vcat(df_list2...)
        set_ret_bp(df_test, ret_interval)
        df_test_map[sb] = df_test
    end
end



############################################################


begin
    target_symbol = "BTCUSDT"
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
    regime_ft_gen_map = Dict("ttc" => (ft_gen_ttc, [0.4, 0.8]), "doi" => (ft_gen_doi, [0.1, 0.5, 0.9]))

    df_evv_train_map::Dict{String, DataFrame} = Dict()
    df_evv_test_map::Dict{String, DataFrame} = Dict()
    for sb in symbols
        println(sb)
        df_evv_train, df_evv_test = get_evv_df_fast2(df_train_map[sb], df_test_map[sb], ft_gen_map, df_train_map[target_symbol].ret_bp, th_vec, is_calc_lr=false)
        # rg_info_map = get_rg_info_map(regime_ft_gen_map, df_train_map[sb], df_test_map[sb], df_evv_train)
        # df_evv_train[!, "regime_weight_sum"] = calc_regime_based_er_vec(rg_info_map, df_evv_train, "mask_train")
        # df_evv_test[!, "regime_weight_sum"] = calc_regime_based_er_vec(rg_info_map, df_evv_test, "mask_test")

        df_evv_train_map[sb] = df_evv_train
        df_evv_test_map[sb] = df_evv_test
    end
end




ft_names = [n for n in names(df_evv_train_map["BTCUSDT"]) if !occursin("_weight_sum", n)]
df_evv_train_multi_sb = DataFrame(timestamp=df_train_map["BTCUSDT"].timestamp)
df_evv_test_multi_sb = DataFrame(timestamp=df_test_map["BTCUSDT"].timestamp)

for sb in symbols
    for ft_name in ft_names
        df_evv_train_multi_sb[!, "$(sb)__$(ft_name)"] = df_evv_train_map[sb][!, ft_name]
        df_evv_test_multi_sb[!, "$(sb)__$(ft_name)"] = df_evv_test_map[sb][!, ft_name]
    end
end


# 지금 rg_info_map 계산이 이상함. SYMBOL__ft_name 으로 계산되어야하는데, SYMBOL이 빠져있음;; 
rg_info_map = get_rg_info_map(regime_ft_gen_map, df_train_map[target_symbol], df_test_map[target_symbol], df_evv_train_map[target_symbol])
df_evv_train_multi_sb[!, "multi_regime_weight_sum"] = calc_multi_regime_based_er_vec(rg_info_map, df_train_multi_sb, "mask_train")
df_evv_test_multi_sb[!, "multi_regime_weight_sum"] = calc_multi_regime_based_er_vec(rg_info_map, df_test_multi_sb, "mask_test")




th = 0.05
th = 0.1
th = 0.33333

begin
    println("Train")
    println("----------")
    sl_train, ss_train = get_signals(df_evv_train_multi_sb[!, "multi_regime_weight_sum"], th)
    plt_train_mrg, tr_res_vec_train_mrg = backtest_sica_2(sl_train, ss_train, df_train_map[target_symbol].WAP_Lag_200ms, df_train_map[target_symbol].WAP_Lag_0ms, df_train_map[target_symbol].timestamp, is_display=false)
    display(plt_train_mrg)
    # final_plt_train = plot(plt_train_eq, plt_train_lr, plt_train_rg, layout=(3, 1), plot_title="Train - th: $th", size=(900, 900))
end





