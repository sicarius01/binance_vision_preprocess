

include("const.jl")
include("utils.jl")
include("feature_generator/feature_generator.jl")

begin
    df_train_map::Dict{String, DataFrame} = Dict()
    @threads for sb in symbols
        date_list_train = [
            # [string(di) for di in 20230313:20230331]; 
            [string(di) for di in 20230401:20230430];
            [string(di) for di in 20230501:20230531]
            ]
        # df_list = [get_df_oneday_full(tardis_dir, s7_dir, sb, date_one_day, features) for date_one_day in date_list_train]
        df_list = [Threads.@spawn get_df_oneday_full(tardis_dir, s7_dir, sb, date, features) for date in date_list_train]
        df_list = fetch.(df_list)

        df_train = vcat(df_list...)
        set_ret_bp(df_train, ret_interval)
        df_train_map[sb] = df_train
    end
end

begin
    df_test_map::Dict{String, DataFrame} = Dict()
    @threads for sb in symbols
        date_list_test = [
            [string(di) for di in 20230525:20230531];
            [string(di) for di in 20230601:20230630]
            ]

        # df_list2 = [get_df_oneday_full(tardis_dir, s7_dir, sb, date_one_day, features) for date_one_day in date_list_test]
        df_list2 = [Threads.@spawn get_df_oneday_full(tardis_dir, s7_dir, sb, date, features) for date in date_list_test]
        df_list2 = fetch.(df_list2)

        df_test = vcat(df_list2...)
        set_ret_bp(df_test, ret_interval)
        df_test_map[sb] = df_test
    end
end



############################################################

thv = [0.05, 0.1, 0.33333, 0.5, 1.0]

all_symbol_plot_train = Dict(th => Dict() for th in thv)
all_symbol_plot_test = Dict(th => Dict() for th in thv)

for target_symbol in symbols
println("target_symbol: $(target_symbol)")
begin
    # target_symbol = "BTCUSDT"
    # target_symbol = "ADAUSDT"
    win = 300
    ret_col_name = "ret_bp"
    # th_vec = [(0.1:0.1:0.9); Float64.(1:99); (99.1:0.1:99.9)]    
    th_vec = [0.01, 0.03, 0.06, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.94, 0.97, 0.99] .* 100
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
        df_evv_train, df_evv_test = get_evv_df_fast(df_train_map[sb], df_test_map[sb], ft_gen_map, df_train_map[target_symbol].ret_bp, th_vec)
        # df_evv_train, df_evv_test = get_evv_df_fast2(df_train_map[sb], df_test_map[sb], ft_gen_map, df_train_map[target_symbol].ret_bp, th_vec, is_calc_lr=false)
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



df_evv_train_multi_sb[!, "multi_equal_weight_sum"] = get_multi_equal_weight_sum(df_evv_train_multi_sb)
df_evv_test_multi_sb[!, "multi_equal_weight_sum"] = get_multi_equal_weight_sum(df_evv_test_multi_sb)

ft_cols = [ft_name for ft_name in names(df_evv_train_multi_sb) if !occursin("weight_sum", ft_name) && ft_name !== "timestamp"]
lrws_train, lrws_test = calc_weighted_sum_return(df_train_map[target_symbol], df_evv_train_multi_sb[!, ft_cols], df_evv_test_multi_sb[!, ft_cols])
df_evv_train_multi_sb[!, "multi_lin_reg_weight_sum"] = lrws_train
df_evv_test_multi_sb[!, "multi_lin_reg_weight_sum"] = lrws_test

rg_info_map = get_rg_info_map(regime_ft_gen_map, df_train_map[target_symbol], df_test_map[target_symbol], df_evv_train_multi_sb)
df_evv_train_multi_sb[!, "multi_regime_weight_sum"] = calc_multi_regime_based_er_vec(rg_info_map, df_evv_train_multi_sb, "mask_train")
df_evv_test_multi_sb[!, "multi_regime_weight_sum"] = calc_multi_regime_based_er_vec(rg_info_map, df_evv_test_multi_sb, "mask_test")




for th in thv
    begin
        println("Train")
        println("----------")
        sl_train, ss_train = get_signals(df_evv_train_multi_sb[!, "multi_equal_weight_sum"], th)
        plt_train_meq, tr_res_vec_train_meq = backtest_sica_2(sl_train, ss_train, df_train_map[target_symbol].WAP_Lag_200ms, df_train_map[target_symbol].WAP_Lag_0ms, df_train_map[target_symbol].timestamp, is_display=false, name=target_symbol)
        
        println("----------")
        sl_train, ss_train = get_signals(df_evv_train_multi_sb[!, "multi_lin_reg_weight_sum"], th)
        plt_train_mlr, tr_res_vec_train_mlr = backtest_sica_2(sl_train, ss_train, df_train_map[target_symbol].WAP_Lag_200ms, df_train_map[target_symbol].WAP_Lag_0ms, df_train_map[target_symbol].timestamp, is_display=false, name=target_symbol)
        
        println("----------")
        sl_train, ss_train = get_signals(df_evv_train_multi_sb[!, "multi_regime_weight_sum"], th)
        plt_train_mrg, tr_res_vec_train_mrg = backtest_sica_2(sl_train, ss_train, df_train_map[target_symbol].WAP_Lag_200ms, df_train_map[target_symbol].WAP_Lag_0ms, df_train_map[target_symbol].timestamp, is_display=false, name=target_symbol)
              
        # final_plt_train = plot(plt_train_meq, plt_train_mlr, plt_train_mrg, layout=(3, 1), plot_title="[$(target_symbol)] Multi Symbol Train - th: $th", size=(900, 900))        
        # savefig(final_plt_train, "./fig/multi_symbol_BT_less_table/$(target_symbol)__multi_symbol_BT_th($(th))_train.png")

        all_symbol_plot_train[th][target_symbol] = plt_train_mlr
    end


    begin
        println("Test")
        println("----------")
        sl_test, ss_test = get_signals(df_evv_test_multi_sb[!, "multi_equal_weight_sum"], th)
        plt_test_meq, tr_res_vec_test_meq = backtest_sica_2(sl_test, ss_test, df_test_map[target_symbol].WAP_Lag_200ms, df_test_map[target_symbol].WAP_Lag_0ms, df_test_map[target_symbol].timestamp, is_display=false, name=target_symbol)
        
        println("----------")
        sl_test, ss_test = get_signals(df_evv_test_multi_sb[!, "multi_lin_reg_weight_sum"], th)
        plt_test_mlr, tr_res_vec_test_mlr = backtest_sica_2(sl_test, ss_test, df_test_map[target_symbol].WAP_Lag_200ms, df_test_map[target_symbol].WAP_Lag_0ms, df_test_map[target_symbol].timestamp, is_display=false, name=target_symbol)
        
        println("----------")
        sl_test, ss_test = get_signals(df_evv_test_multi_sb[!, "multi_regime_weight_sum"], th)
        plt_test_mrg, tr_res_vec_test_mrg = backtest_sica_2(sl_test, ss_test, df_test_map[target_symbol].WAP_Lag_200ms, df_test_map[target_symbol].WAP_Lag_0ms, df_test_map[target_symbol].timestamp, is_display=false, name=target_symbol)
        
        # final_plt_test = plot(plt_test_meq, plt_test_mlr, plt_test_mrg, layout=(3, 1), plot_title="[$(target_symbol)] Multi Symbol Test - th: $th", size=(900, 900))        
        # savefig(final_plt_test, "./fig/multi_symbol_BT_less_table/$(target_symbol)__multi_symbol_BT_th($(th))_test.png")

        all_symbol_plot_test[th][target_symbol] = plt_test_mlr
    end
end

end     # symbols로 target_symbol 돌리는 for loop


all_symbol_plot_train = plot(title="Multi Symbol Linear Regression")


for th in thv
    symbols_plot_train = plot([all_symbol_plot_train[th][sb] for sb in symbols]..., layout=(3, 2), plot_title="Multi Symbol Lin Reg th: $(th) [Train]", size=(2000, 1800))
    savefig(symbols_plot_train, "./fig/summary/multi_symbol_lin_reg_train_th($(th)).png")

    symbols_plot_test = plot([all_symbol_plot_test[th][sb] for sb in symbols]..., layout=(3, 2), plot_title="Multi Symbol Lin Reg th: $(th) [Test]", size=(2000, 1800))
    savefig(symbols_plot_test, "./fig/summary/multi_symbol_lin_reg_test_th($(th)).png")

end




# 투두 완료
# 1. 48개 피쳐 equal weight, lr 해서 비교
# 2. 심볼 별로 피쳐 깎아서 해보기







# th = 0.05
# th = 0.1
# th = 0.33333
# th = 0.5
# th = 1.0
# th = 5.0


# begin
#     println("Train")
#     println("----------")
#     sl_train, ss_train = get_signals(df_evv_train_multi_sb[!, "multi_regime_weight_sum"], th)
#     plt_train_mrg, tr_res_vec_train_mrg = backtest_sica_2(sl_train, ss_train, df_train_map[target_symbol].WAP_Lag_200ms, df_train_map[target_symbol].WAP_Lag_0ms, df_train_map[target_symbol].timestamp, is_display=false)
#     display(plt_train_mrg)
#     # final_plt_train = plot(plt_train_eq, plt_train_lr, plt_train_rg, layout=(3, 1), plot_title="Train - th: $th", size=(900, 900))
# end


# begin
#     println("Test")
#     println("----------")
#     sl_test, ss_test = get_signals(df_evv_test_multi_sb[!, "multi_regime_weight_sum"], th)
#     plt_test_mrg, tr_res_vec_test_mrg = backtest_sica_2(sl_test, ss_test, df_test_map[target_symbol].WAP_Lag_200ms, df_test_map[target_symbol].WAP_Lag_0ms, df_test_map[target_symbol].timestamp, is_display=false)
#     display(plt_test_mrg)
#     # final_plt_train = plot(plt_train_eq, plt_train_lr, plt_train_rg, layout=(3, 1), plot_title="Train - th: $th", size=(900, 900))
# end



# ####################################
# # 상위 10개 피쳐만 사용해서 다시 해보자

# use_features = [
#     "BTCUSDT__ofi", "BTCUSDT__tv", "BTCUSDT__liquidation", "BTCUSDT__mark_dist", "BTCUSDT__index_dist", "BTCUSDT__tor", "BTCUSDT__aq", "BTCUSDT__vwap",
#     "SOLUSDT__vwap",
#     "SOLUSDT__mark_dist",
#     "SOLUSDT__liquidation",
#     # "BTCUSDT__liquidation",
#     # "BTCUSDT__ofi",
#     "DOGEUSDT__tor",
#     "XRPUSDT__tor",
#     "ETHUSDT__aq",
#     "SOLUSDT__aq",
#     "SOLUSDT__ofi",
# ]


# df_evv_train_filtered = df_evv_train_multi_sb[:, use_features]
# df_evv_test_filtered = df_evv_test_multi_sb[:, use_features]

# rg_info_map = get_rg_info_map(regime_ft_gen_map, df_train_map[target_symbol], df_test_map[target_symbol], df_evv_train_filtered)

# df_evv_train_multi_sb[!, "multi_regime_10_weight_sum"] = calc_multi_regime_based_er_vec(rg_info_map, df_evv_train_filtered, "mask_train")
# df_evv_test_multi_sb[!, "multi_regime_10_weight_sum"] = calc_multi_regime_based_er_vec(rg_info_map, df_evv_test_filtered, "mask_test")


# th = 0.05
# th = 0.1
# th = 0.33333
# th = 0.5
# th = 1.0
# th = 5.0

# thv = [0.05, 0.1, 0.33333, 0.5, 1.0]
# for th in thv
#     begin
#         println("Train")
#         println("----------")
#         sl_train, ss_train = get_signals(df_evv_train_multi_sb[!, "multi_regime_weight_sum"], th)
#         plt_train_mrg, tr_res_vec_train_mrg = backtest_sica_2(sl_train, ss_train, df_train_map[target_symbol].WAP_Lag_200ms, df_train_map[target_symbol].WAP_Lag_0ms, df_train_map[target_symbol].timestamp, is_display=false)
        
#         println("----------")
#         sl_train, ss_train = get_signals(df_evv_train_multi_sb[!, "multi_regime_10_weight_sum"], th)
#         plt_train_mrg_10, tr_res_vec_train_mrg_10 = backtest_sica_2(sl_train, ss_train, df_train_map[target_symbol].WAP_Lag_200ms, df_train_map[target_symbol].WAP_Lag_0ms, df_train_map[target_symbol].timestamp, is_display=false)
#         # display(plt_train_mrg)
#         final_plt_train = plot(plt_train_mrg, plt_train_mrg_10, layout=(2, 1), plot_title="[$(target_symbol)] Multi Symbol Train - th: $th", size=(900, 900))
#         # final_plt_train = plot(plt_train_mrg, layout=(1, 1), plot_title="[$(target_symbol)] Multi Symbol Train - th: $th", size=(900, 900))
#         savefig(final_plt_train, "./fig/$(target_symbol)__multi_symbol_BT_th($(th))_train2.png")
#     end


#     begin
#         println("Test")
#         println("----------")
#         sl_test, ss_test = get_signals(df_evv_test_multi_sb[!, "multi_regime_weight_sum"], th)
#         plt_test_mrg, tr_res_vec_test_mrg = backtest_sica_2(sl_test, ss_test, df_test_map[target_symbol].WAP_Lag_200ms, df_test_map[target_symbol].WAP_Lag_0ms, df_test_map[target_symbol].timestamp, is_display=false)
        
#         println("----------")
#         sl_test, ss_test = get_signals(df_evv_test_multi_sb[!, "multi_regime_10_weight_sum"], th)
#         plt_test_mrg_10, tr_res_vec_test_mrg_10 = backtest_sica_2(sl_test, ss_test, df_test_map[target_symbol].WAP_Lag_200ms, df_test_map[target_symbol].WAP_Lag_0ms, df_test_map[target_symbol].timestamp, is_display=false)
#         # display(plt_test_mrg)
#         final_plt_test = plot(plt_test_mrg, plt_test_mrg_10, layout=(2, 1), plot_title="[$(target_symbol)] Multi Symbol Test - th: $th", size=(900, 900))
#         # final_plt_test = plot(plt_test_mrg, layout=(1, 1), plot_title="[$(target_symbol)] Multi Symbol Test - th: $th", size=(900, 900))
#         savefig(final_plt_test, "./fig/$(target_symbol)__multi_symbol_BT_th($(th))_test2.png")
#     end
# end
# end



# 결론
# 48개 피쳐가 직교는 한다.
# 근데 합치면 성과가 영 별로임 
# 그냥 1개 종목에서 8개 뽑은걸로 쓰자 
# 1개 종목에서 다 잘 통하는지나 한번 보자.... 










# ################################
# # 성능이 구림. 일단 48개 피쳐 직교성을 좀 보자

# # 추가 패키지 import (직교성 분석용)
# using LinearAlgebra
# using Statistics  
# using StatsBase
# using Printf

# # corr_analysis.jl의 함수들을 여기에 정의
# """
# 피어슨 상관관계 행렬을 계산하고 시각화하는 함수
# """
# function analyze_feature_correlation(df_evv_train, df_evv_test, ft_names; title_suffix="")
#     println("\n" * "="^60)
#     println("Feature Correlation Analysis $(title_suffix)")
#     println("="^60)
    
#     # Train 데이터 상관관계
#     train_data = Matrix(df_evv_train[!, ft_names])
#     # NaN 값이 있는 행 제거
#     valid_rows = .!any(isnan.(train_data), dims=2)[:]
#     train_clean = train_data[valid_rows, :]
    
#     corr_train = cor(train_clean)
    
#     # Test 데이터 상관관계  
#     test_data = Matrix(df_evv_test[!, ft_names])
#     valid_rows_test = .!any(isnan.(test_data), dims=2)[:]
#     test_clean = test_data[valid_rows_test, :]
    
#     corr_test = cor(test_clean)
    
#     # 상관관계 통계 출력
#     println("Train 데이터:")
#     print_correlation_stats(corr_train, ft_names)
#     println("\nTest 데이터:")
#     print_correlation_stats(corr_test, ft_names)
    
#     # 히트맵 시각화
#     p1 = heatmap(corr_train, 
#                 xticks=(1:length(ft_names), ft_names), 
#                 yticks=(1:length(ft_names), ft_names),
#                 title="Train Correlation Matrix",
#                 color=:RdBu,
#                 clims=(-1, 1),
#                 xrotation=45)
    
#     p2 = heatmap(corr_test, 
#                 xticks=(1:length(ft_names), ft_names), 
#                 yticks=(1:length(ft_names), ft_names),
#                 title="Test Correlation Matrix", 
#                 color=:RdBu,
#                 clims=(-1, 1),
#                 xrotation=45)
    
#     plt_corr = plot(p1, p2, layout=(1, 2), size=(1400, 600))
#     display(plt_corr)
    
#     return corr_train, corr_test, plt_corr
# end

# """
# 상관관계 통계 출력 함수
# """
# function print_correlation_stats(corr_matrix, ft_names)
#     n = size(corr_matrix, 1)
    
#     # 대각선 제외한 상관계수들 추출
#     off_diag_corrs = []
#     for i in 1:n
#         for j in (i+1):n
#             push!(off_diag_corrs, abs(corr_matrix[i, j]))
#         end
#     end
    
#     println(@sprintf("  평균 절대 상관계수: %.4f", mean(off_diag_corrs)))
#     println(@sprintf("  최대 절대 상관계수: %.4f", maximum(off_diag_corrs)))
#     println(@sprintf("  상관계수 > 0.5인 쌍: %d개 (%.1f%%)", 
#              sum(off_diag_corrs .> 0.5), 
#              100 * sum(off_diag_corrs .> 0.5) / length(off_diag_corrs)))
#     println(@sprintf("  상관계수 > 0.7인 쌍: %d개 (%.1f%%)", 
#              sum(off_diag_corrs .> 0.7), 
#              100 * sum(off_diag_corrs .> 0.7) / length(off_diag_corrs)))
    
#     # 높은 상관관계 쌍들 출력
#     high_corr_pairs = []
#     for i in 1:n
#         for j in (i+1):n
#             if abs(corr_matrix[i, j]) > 0.5
#                 push!(high_corr_pairs, (ft_names[i], ft_names[j], corr_matrix[i, j]))
#             end
#         end
#     end
    
#     if !isempty(high_corr_pairs)
#         println("  높은 상관관계 (|r| > 0.5):")
#         for (ft1, ft2, corr_val) in sort(high_corr_pairs, by=x->abs(x[3]), rev=true)
#             println(@sprintf("    %s - %s: %.4f", ft1, ft2, corr_val))
#         end
#     end
# end

# """
# PCA 분석 함수
# """
# function analyze_feature_pca(df_evv_train, df_evv_test, ft_names; title_suffix="")
#     println("\n" * "="^60)
#     println("Principal Component Analysis (PCA) $(title_suffix)")
#     println("="^60)
    
#     # Train 데이터 준비
#     train_data = Matrix(df_evv_train[!, ft_names])
#     valid_rows = .!any(isnan.(train_data), dims=2)[:]
#     train_clean = train_data[valid_rows, :]
    
#     # 데이터 표준화
#     train_std = StatsBase.standardize(StatsBase.ZScoreTransform, train_clean, dims=1)
    
#     # PCA 수행 (SVD 사용)
#     U, S, Vt = svd(train_std)
    
#     # 설명된 분산 비율 계산
#     explained_variance_ratio = (S .^ 2) ./ sum(S .^ 2)
#     cumulative_variance = cumsum(explained_variance_ratio)
    
#     # 결과 출력
#     println("주성분별 설명된 분산:")
#     for i in 1:min(length(S), 10)  # 최대 10개까지만 출력
#         println(@sprintf("  PC%d: %.4f (누적: %.4f)", 
#                 i, explained_variance_ratio[i], cumulative_variance[i]))
#     end
    
#     # 90% 분산을 설명하는 주성분 개수
#     n_components_90 = findfirst(cumulative_variance .>= 0.9)
#     n_components_95 = findfirst(cumulative_variance .>= 0.95)
#     println(@sprintf("\n90%% 분산 설명에 필요한 주성분: %d개", n_components_90))
#     println(@sprintf("95%% 분산 설명에 필요한 주성분: %d개", n_components_95))
    
#     # 첫 번째 주성분의 feature loading 출력
#     println("\n첫 번째 주성분 (PC1) 로딩 (절댓값 기준 상위 10개):")
#     pc1_loadings = Vt[1, :]
#     loading_pairs = [(abs(pc1_loadings[i]), ft_names[i], pc1_loadings[i]) for i in 1:length(ft_names)]
#     sort!(loading_pairs, rev=true)
#     for i in 1:min(10, length(loading_pairs))
#         println(@sprintf("  %s: %.4f", loading_pairs[i][2], loading_pairs[i][3]))
#     end
    
#     # 시각화
#     p1 = plot(1:min(length(explained_variance_ratio), 15), 
#               explained_variance_ratio[1:min(length(explained_variance_ratio), 15)], 
#               seriestype=:bar, 
#               title="Explained Variance Ratio", 
#               xlabel="Principal Component", 
#               ylabel="Variance Ratio",
#               legend=false)
    
#     p2 = plot(1:min(length(cumulative_variance), 15), 
#               cumulative_variance[1:min(length(cumulative_variance), 15)], 
#               title="Cumulative Explained Variance", 
#               xlabel="Principal Component", 
#               ylabel="Cumulative Variance Ratio",
#               legend=false,
#               linewidth=2)
#     hline!(p2, [0.9, 0.95], linestyle=:dash, color=:red, alpha=0.5)
    
#     plt_pca = plot(p1, p2, layout=(1, 2), size=(1200, 400))
#     display(plt_pca)
    
#     return explained_variance_ratio, cumulative_variance, Vt, plt_pca
# end

# """
# 직교성 종합 분석 함수
# """
# function comprehensive_orthogonality_analysis(df_evv_train, df_evv_test, ft_names)
#     println("\n" * "="^80)
#     println("피처 직교성 종합 분석")
#     println("="^80)
    
#     # 1. 상관관계 분석
#     corr_train, corr_test, plt_corr = analyze_feature_correlation(df_evv_train, df_evv_test, ft_names)
    
#     # 2. PCA 분석
#     explained_var, cumulative_var, loadings, plt_pca = analyze_feature_pca(df_evv_train, df_evv_test, ft_names)
    
#     # 3. 직교성 점수 계산
#     println("\n" * "="^60)
#     println("직교성 종합 점수")
#     println("="^60)
    
#     # Train 데이터 기준 직교성 점수
#     n = length(ft_names)
#     off_diag_corrs = []
#     for i in 1:n
#         for j in (i+1):n
#             push!(off_diag_corrs, abs(corr_train[i, j]))
#         end
#     end
    
#     orthogonality_score = 1.0 - mean(off_diag_corrs)  # 평균 절대 상관계수가 낮을수록 직교성 높음
#     pca_score = 1.0 - explained_var[1]  # 첫 번째 주성분이 설명하는 분산이 낮을수록 분산이 잘 분산됨
    
#     println(@sprintf("평균 절대 상관계수 기반 직교성 점수: %.4f (1에 가까울수록 직교)", orthogonality_score))
#     println(@sprintf("PCA 기반 분산 분산도 점수: %.4f (1에 가까울수록 분산이 고르게 분포)", pca_score))
    
#     # 4. 권장사항 출력
#     println("\n권장사항:")
#     if mean(off_diag_corrs) > 0.3
#         println("⚠️  피처들 간 상관관계가 높습니다. 차원 축소를 고려해보세요.")
#     else
#         println("✅ 피처들이 적절히 독립적입니다.")
#     end
    
#     if explained_var[1] > 0.4
#         println("⚠️  첫 번째 주성분이 분산의 40% 이상을 설명합니다. 피처 다양성을 늘려보세요.")
#     else
#         println("✅ 분산이 주성분들에 고르게 분포되어 있습니다.")
#     end
    
#     n_components_95 = findfirst(cumulative_var .>= 0.95)
#     efficiency = n_components_95 / length(ft_names)
#     if efficiency < 0.7
#         println(@sprintf("✅ 차원 효율성이 좋습니다. %d개 피처로 95%% 분산 설명 가능 (효율성: %.2f)", n_components_95, efficiency))
#     else
#         println(@sprintf("⚠️  차원 효율성이 낮습니다. %d개 피처가 95%% 분산 설명에 필요 (효율성: %.2f)", n_components_95, efficiency))
#     end
    
#     return Dict(
#         "correlation_matrices" => (corr_train, corr_test),
#         "pca_results" => (explained_var, cumulative_var, loadings),
#         "orthogonality_score" => orthogonality_score,
#         "pca_score" => pca_score,
#         "plots" => (plt_corr, plt_pca)
#     )
# end

# # 48개 피쳐 직교성 분석 실행
# begin
#     println("\n🔍 Multi-Symbol 48개 피쳐 직교성 분석을 시작합니다...")
    
#     # timestamp와 multi_regime_weight_sum 제외한 48개 피쳐 추출
#     all_columns = names(df_evv_train_multi_sb)
#     exclude_columns = ["timestamp", "multi_regime_weight_sum"]
#     multi_ft_names = [col for col in all_columns if !(col in exclude_columns)]
    
#     println("분석 대상 피쳐 수: $(length(multi_ft_names))개")
#     println("피쳐 목록:")
#     for (i, ft_name) in enumerate(multi_ft_names)
#         println("  $i. $ft_name")
#     end
    
#     # 종합 직교성 분석 실행
#     multi_orthogonality_results = comprehensive_orthogonality_analysis(
#         df_evv_train_multi_sb, df_evv_test_multi_sb, multi_ft_names)
    
#     println("\n" * "="^80)
#     println("🎯 Multi-Symbol 48개 피쳐 분석 요약")
#     println("="^80)
#     println(@sprintf("전체 피처 수: %d개", length(multi_ft_names)))
#     println(@sprintf("직교성 점수: %.4f", multi_orthogonality_results["orthogonality_score"]))
#     println(@sprintf("PCA 분산 분산도: %.4f", multi_orthogonality_results["pca_score"]))
    
#     # Symbol별, Feature별 상관관계 패턴 분석
#     println("\n📊 Symbol별 피쳐 그룹 분석:")
#     symbol_groups = Dict()
#     for ft_name in multi_ft_names
#         if occursin("__", ft_name)
#             symbol, feature = split(ft_name, "__", limit=2)
#             if !haskey(symbol_groups, symbol)
#                 symbol_groups[symbol] = []
#             end
#             push!(symbol_groups[symbol], ft_name)
#         end
#     end
    
#     for (symbol, features) in symbol_groups
#         println("  $symbol: $(length(features))개 피쳐")
#     end
    
#     # 동일 피쳐 타입의 Symbol간 상관관계 분석
#     println("\n📈 동일 피쳐 타입의 Symbol간 상관관계 분석:")
#     corr_train = multi_orthogonality_results["correlation_matrices"][1]
    
#     # 피쳐 타입별로 그룹핑
#     feature_types = Dict()
#     for ft_name in multi_ft_names
#         if occursin("__", ft_name)
#             symbol, feature = split(ft_name, "__", limit=2)
#             if !haskey(feature_types, feature)
#                 feature_types[feature] = []
#             end
#             push!(feature_types[feature], ft_name)
#         end
#     end
    
#     for (feature_type, feature_list) in feature_types
#         if length(feature_list) > 1
#             # 동일 피쳐 타입 내에서의 평균 상관계수 계산
#             type_corrs = []
#             feature_indices = [findfirst(x -> x == ft, multi_ft_names) for ft in feature_list]
#             for i in 1:length(feature_indices)
#                 for j in (i+1):length(feature_indices)
#                     idx1, idx2 = feature_indices[i], feature_indices[j]
#                     push!(type_corrs, abs(corr_train[idx1, idx2]))
#                 end
#             end
#             if !isempty(type_corrs)
#                 println(@sprintf("  %s: 평균 절대 상관계수 %.4f (Symbol간)", 
#                         feature_type, mean(type_corrs)))
#             end
#         end
#     end
# end












