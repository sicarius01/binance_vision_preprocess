


include("const.jl")
include("utils.jl")
include("feature_generator/feature_generator.jl")

using LinearAlgebra
using Statistics
using StatsBase
using Plots
using Printf

date_list_train = [
    # [string(di) for di in 20230313:20230331]; 
    [string(di) for di in 20230401:20230430];
    [string(di) for di in 20230501:20230531]
    ]

df_list = [get_df_oneday_full(tardis_dir, s7_dir, symbol, date_one_day, features) for date_one_day in date_list_train]
df_train = vcat(df_list...)
set_ret_bp(df_train, ret_interval)


date_list_test = [
    [string(di) for di in 20230525:20230531];
    [string(di) for di in 20230601:20230630]
    ]

df_list2 = [get_df_oneday_full(tardis_dir, s7_dir, symbol, date_one_day, features) for date_one_day in date_list_test]
df_test = vcat(df_list2...)
set_ret_bp(df_test, ret_interval)




ret_col_name = "ret_bp"
th_vec = [(0.1:0.1:0.9); Float64.(1:99); (99.1:0.1:99.9)]

println("--------------------------------------------\n")
ft_gen_map = Dict(
    "ofi" => ft_gen_ofi,
    "aq" => ft_gen_aq,
    "tor" => ft_gen_tor,
    "tv" => ft_gen_tv,
    "index_dist" => ft_gen_index_dist,
    "mark_dist" => ft_gen_mark_dist,
    "trade_cnt_ratio" => ft_gen_trcr,
    # "trade_vol" => ft_gen_trv,
    "liquidation" => ft_gen_liqu,
    "vwap" => ft_gen_vwap,
)
ft_names = collect(keys(ft_gen_map))

df_evv_train, df_evv_test = get_evv_df_fast(ft_gen_map, ret_col_name, th_vec)
for th in [0.025, 0.05, 0.1, 0.5, 1.0, 5.0]
    summary_evv_result(df_evv_train, df_evv_test, ft_names, th)
end



# df_evv_train

# df_evv_test


# ========================================
# 피처 직교성 분석 함수들
# ========================================

begin
"""
피어슨 상관관계 행렬을 계산하고 시각화하는 함수
"""
function analyze_feature_correlation(df_evv_train, df_evv_test, ft_names; title_suffix="")
    println("\n" * "="^60)
    println("Feature Correlation Analysis $(title_suffix)")
    println("="^60)
    
    # Train 데이터 상관관계
    train_data = Matrix(df_evv_train[!, ft_names])
    # NaN 값이 있는 행 제거
    valid_rows = .!any(isnan.(train_data), dims=2)[:]
    train_clean = train_data[valid_rows, :]
    
    corr_train = cor(train_clean)
    
    # Test 데이터 상관관계  
    test_data = Matrix(df_evv_test[!, ft_names])
    valid_rows_test = .!any(isnan.(test_data), dims=2)[:]
    test_clean = test_data[valid_rows_test, :]
    
    corr_test = cor(test_clean)
    
    # 상관관계 통계 출력
    println("Train 데이터:")
    print_correlation_stats(corr_train, ft_names)
    println("\nTest 데이터:")
    print_correlation_stats(corr_test, ft_names)
    
    # 히트맵 시각화
    p1 = heatmap(corr_train, 
                xticks=(1:length(ft_names), ft_names), 
                yticks=(1:length(ft_names), ft_names),
                title="Train Correlation Matrix",
                color=:RdBu,
                clims=(-1, 1))
    
    p2 = heatmap(corr_test, 
                xticks=(1:length(ft_names), ft_names), 
                yticks=(1:length(ft_names), ft_names),
                title="Test Correlation Matrix", 
                color=:RdBu,
                clims=(-1, 1))
    
    plt_corr = plot(p1, p2, layout=(1, 2), size=(1200, 500))
    display(plt_corr)
    
    return corr_train, corr_test, plt_corr
end

"""
상관관계 통계 출력 함수
"""
function print_correlation_stats(corr_matrix, ft_names)
    n = size(corr_matrix, 1)
    
    # 대각선 제외한 상관계수들 추출
    off_diag_corrs = []
    for i in 1:n
        for j in (i+1):n
            push!(off_diag_corrs, abs(corr_matrix[i, j]))
        end
    end
    
    println(@sprintf("  평균 절대 상관계수: %.4f", mean(off_diag_corrs)))
    println(@sprintf("  최대 절대 상관계수: %.4f", maximum(off_diag_corrs)))
    println(@sprintf("  상관계수 > 0.5인 쌍: %d개 (%.1f%%)", 
             sum(off_diag_corrs .> 0.5), 
             100 * sum(off_diag_corrs .> 0.5) / length(off_diag_corrs)))
    println(@sprintf("  상관계수 > 0.7인 쌍: %d개 (%.1f%%)", 
             sum(off_diag_corrs .> 0.7), 
             100 * sum(off_diag_corrs .> 0.7) / length(off_diag_corrs)))
    
    # 높은 상관관계 쌍들 출력
    high_corr_pairs = []
    for i in 1:n
        for j in (i+1):n
            if abs(corr_matrix[i, j]) > 0.5
                push!(high_corr_pairs, (ft_names[i], ft_names[j], corr_matrix[i, j]))
            end
        end
    end
    
    if !isempty(high_corr_pairs)
        println("  높은 상관관계 (|r| > 0.5):")
        for (ft1, ft2, corr_val) in sort(high_corr_pairs, by=x->abs(x[3]), rev=true)
            println(@sprintf("    %s - %s: %.4f", ft1, ft2, corr_val))
        end
    end
end

"""
PCA 분석 함수
"""
function analyze_feature_pca(df_evv_train, df_evv_test, ft_names; title_suffix="")
    println("\n" * "="^60)
    println("Principal Component Analysis (PCA) $(title_suffix)")
    println("="^60)
    
    # Train 데이터 준비
    train_data = Matrix(df_evv_train[!, ft_names])
    valid_rows = .!any(isnan.(train_data), dims=2)[:]
    train_clean = train_data[valid_rows, :]
    
    # 데이터 표준화
    train_std = StatsBase.standardize(StatsBase.ZScoreTransform, train_clean, dims=1)
    
    # PCA 수행 (SVD 사용)
    U, S, Vt = svd(train_std)
    
    # 설명된 분산 비율 계산
    explained_variance_ratio = (S .^ 2) ./ sum(S .^ 2)
    cumulative_variance = cumsum(explained_variance_ratio)
    
    # 결과 출력
    println("주성분별 설명된 분산:")
    for i in 1:min(length(S), 10)  # 최대 10개까지만 출력
        println(@sprintf("  PC%d: %.4f (누적: %.4f)", 
                i, explained_variance_ratio[i], cumulative_variance[i]))
    end
    
    # 90% 분산을 설명하는 주성분 개수
    n_components_90 = findfirst(cumulative_variance .>= 0.9)
    n_components_95 = findfirst(cumulative_variance .>= 0.95)
    println(@sprintf("\n90%% 분산 설명에 필요한 주성분: %d개", n_components_90))
    println(@sprintf("95%% 분산 설명에 필요한 주성분: %d개", n_components_95))
    
    # 첫 번째 주성분의 feature loading 출력
    println("\n첫 번째 주성분 (PC1) 로딩:")
    pc1_loadings = Vt[1, :]
    for (i, ft_name) in enumerate(ft_names)
        println(@sprintf("  %s: %.4f", ft_name, pc1_loadings[i]))
    end
    
    # 시각화
    p1 = plot(1:min(length(explained_variance_ratio), 15), 
              explained_variance_ratio[1:min(length(explained_variance_ratio), 15)], 
              seriestype=:bar, 
              title="Explained Variance Ratio", 
              xlabel="Principal Component", 
              ylabel="Variance Ratio",
              legend=false)
    
    p2 = plot(1:min(length(cumulative_variance), 15), 
              cumulative_variance[1:min(length(cumulative_variance), 15)], 
              title="Cumulative Explained Variance", 
              xlabel="Principal Component", 
              ylabel="Cumulative Variance Ratio",
              legend=false,
              linewidth=2)
    hline!(p2, [0.9, 0.95], linestyle=:dash, color=:red, alpha=0.5)
    
    # Feature loading 히트맵 (첫 5개 주성분)
    n_show = min(5, size(Vt, 1))
    p3 = heatmap(Vt[1:n_show, :], 
                xticks=(1:length(ft_names), ft_names), 
                yticks=(1:n_show, ["PC$i" for i in 1:n_show]),
                title="Principal Component Loadings",
                color=:RdBu,
                clims=(-1, 1))
    
    plt_pca = plot(p1, p2, p3, layout=(2, 2), size=(1200, 800))
    display(plt_pca)
    
    return explained_variance_ratio, cumulative_variance, Vt, plt_pca
end

"""
직교성 종합 분석 함수
"""
function comprehensive_orthogonality_analysis(df_evv_train, df_evv_test, ft_names)
    println("\n" * "="^80)
    println("피처 직교성 종합 분석")
    println("="^80)
    
    # 1. 상관관계 분석
    corr_train, corr_test, plt_corr = analyze_feature_correlation(df_evv_train, df_evv_test, ft_names)
    
    # 2. PCA 분석
    explained_var, cumulative_var, loadings, plt_pca = analyze_feature_pca(df_evv_train, df_evv_test, ft_names)
    
    # 3. 직교성 점수 계산
    println("\n" * "="^60)
    println("직교성 종합 점수")
    println("="^60)
    
    # Train 데이터 기준 직교성 점수
    n = length(ft_names)
    off_diag_corrs = []
    for i in 1:n
        for j in (i+1):n
            push!(off_diag_corrs, abs(corr_train[i, j]))
        end
    end
    
    orthogonality_score = 1.0 - mean(off_diag_corrs)  # 평균 절대 상관계수가 낮을수록 직교성 높음
    pca_score = 1.0 - explained_var[1]  # 첫 번째 주성분이 설명하는 분산이 낮을수록 분산이 잘 분산됨
    
    println(@sprintf("평균 절대 상관계수 기반 직교성 점수: %.4f (1에 가까울수록 직교)", orthogonality_score))
    println(@sprintf("PCA 기반 분산 분산도 점수: %.4f (1에 가까울수록 분산이 고르게 분포)", pca_score))
    
    # 4. 권장사항 출력
    println("\n권장사항:")
    if mean(off_diag_corrs) > 0.3
        println("⚠️  피처들 간 상관관계가 높습니다. 차원 축소를 고려해보세요.")
    else
        println("✅ 피처들이 적절히 독립적입니다.")
    end
    
    if explained_var[1] > 0.4
        println("⚠️  첫 번째 주성분이 분산의 40% 이상을 설명합니다. 피처 다양성을 늘려보세요.")
    else
        println("✅ 분산이 주성분들에 고르게 분포되어 있습니다.")
    end
    
    n_components_95 = findfirst(cumulative_var .>= 0.95)
    efficiency = n_components_95 / length(ft_names)
    if efficiency < 0.7
        println(@sprintf("✅ 차원 효율성이 좋습니다. %d개 피처로 95%% 분산 설명 가능 (효율성: %.2f)", n_components_95, efficiency))
    else
        println(@sprintf("⚠️  차원 효율성이 낮습니다. %d개 피처가 95%% 분산 설명에 필요 (효율성: %.2f)", n_components_95, efficiency))
    end
    
    return Dict(
        "correlation_matrices" => (corr_train, corr_test),
        "pca_results" => (explained_var, cumulative_var, loadings),
        "orthogonality_score" => orthogonality_score,
        "pca_score" => pca_score,
        "plots" => (plt_corr, plt_pca)
    )
end


# ========================================
# 직교성 분석 실행
# ========================================

"""
피처 클러스터링 분석 (유사한 피처들 그룹핑)
"""
function analyze_feature_clustering(corr_matrix, ft_names; linkage_method=:ward, n_clusters=3)
    println("\n" * "="^60)
    println("피처 클러스터링 분석")
    println("="^60)
    
    # 거리 행렬 계산 (1 - |correlation|)
    dist_matrix = 1.0 .- abs.(corr_matrix)
    
    # 계층적 클러스터링 수행 (간단한 구현)
    n = length(ft_names)
    clusters = collect(1:n)  # 각 피처를 별도 클러스터로 시작
    
    # 가장 가까운 피처들 찾기
    distances = []
    for i in 1:n
        for j in (i+1):n
            push!(distances, (dist_matrix[i, j], i, j))
        end
    end
    sort!(distances)
    
    println("피처 간 거리 (작을수록 유사):")
    for (dist, i, j) in distances[1:min(10, length(distances))]
        println(@sprintf("  %s - %s: %.4f (상관계수: %.4f)", 
                ft_names[i], ft_names[j], dist, corr_matrix[i, j]))
    end
    
    # 가장 유사한 피처 쌍들 식별
    similar_pairs = [(dist, i, j) for (dist, i, j) in distances if dist < 0.5]  # 상관계수 0.5 이상
    
    if !isempty(similar_pairs)
        println("\n중복성이 높은 피처 쌍들 (거리 < 0.5, |상관계수| > 0.5):")
        for (dist, i, j) in similar_pairs
            println(@sprintf("  %s ↔ %s (거리: %.3f, 상관: %.3f)", 
                    ft_names[i], ft_names[j], dist, corr_matrix[i, j]))
        end
    else
        println("\n✅ 중복성이 높은 피처 쌍이 발견되지 않았습니다.")
    end
    
    return similar_pairs
end

"""
피처 선택 권장 함수
"""
function recommend_feature_selection(corr_matrix, explained_var, ft_names; 
                                   corr_threshold=0.7, pca_importance_threshold=0.1)
    println("\n" * "="^60)
    println("피처 선택 권장사항")
    println("="^60)
    
    n = length(ft_names)
    
    # 1. 높은 상관관계를 가진 피처들 식별
    high_corr_pairs = []
    for i in 1:n
        for j in (i+1):n
            if abs(corr_matrix[i, j]) > corr_threshold
                push!(high_corr_pairs, (i, j, corr_matrix[i, j]))
            end
        end
    end
    
    # 2. PCA 첫 번째 주성분에서 중요도가 낮은 피처들 식별
    pc1_loadings = abs.(explained_var)  # 이미 loadings가 아니라 explained variance ratio
    
    println("피처 중복성 분석:")
    if !isempty(high_corr_pairs)
        println("⚠️  높은 상관관계 피처 쌍들:")
        for (i, j, corr_val) in high_corr_pairs
            println(@sprintf("  %s - %s: %.4f", ft_names[i], ft_names[j], corr_val))
        end
        
        # 중복 피처 제거 권장
        redundant_features = Set()
        for (i, j, corr_val) in high_corr_pairs
            # 더 적은 정보를 가진 피처를 제거 후보로 선택 (임의로 j 선택)
            push!(redundant_features, j)
        end
        
        if !isempty(redundant_features)
            println("\n제거 고려 대상 피처들:")
            for idx in redundant_features
                println(@sprintf("  - %s", ft_names[idx]))
            end
        end
    else
        println("✅ 높은 상관관계를 가진 피처 쌍이 없습니다.")
    end
    
    # 3. 최종 권장 피처 세트
    recommended_features = setdiff(1:n, Set(j for (i, j, _) in high_corr_pairs))
    
    println("\n추천 피처 세트:")
    for idx in recommended_features
        println(@sprintf("  ✓ %s", ft_names[idx]))
    end
    
    println(@sprintf("\n요약: %d개 피처 중 %d개 추천 (%.1f%% 유지)", 
            n, length(recommended_features), 100 * length(recommended_features) / n))
    
    return collect(recommended_features), [ft_names[i] for i in recommended_features]
end

"""
피처 중요도 vs 직교성 트레이드오프 분석
"""
function analyze_importance_orthogonality_tradeoff(df_evv_train, df_evv_test, ft_names, 
                                                  ret_col_name="ret_bp"; test_threshold=5.0)
    println("\n" * "="^60)
    println("피처 중요도 vs 직교성 트레이드오프 분석")
    println("="^60)
    
    # 각 피처의 개별 성능 측정
    individual_performances = []
    for ft_name in ft_names
        # 간단한 성능 측정: 해당 피처만 사용했을 때 신호 성능
        sl_train, ss_train = get_signals(df_evv_train[!, ft_name], test_threshold)
        plt_train, tr_res_vec_train = backtest_sica_2(sl_train, ss_train, 
                                                     df_train.WAP_Lag_200ms, 
                                                     df_train.WAP_Lag_0ms, 
                                                     df_train.timestamp, is_print=false)
        
        if !isempty(tr_res_vec_train)
            avg_return = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_train])
            push!(individual_performances, (ft_name, avg_return, length(tr_res_vec_train)))
        else
            push!(individual_performances, (ft_name, 0.0, 0))
        end
    end
    
    # 성능 순으로 정렬
    sort!(individual_performances, by=x->x[2], rev=true)
    
    println("개별 피처 성능 (평균 수익률 bp):")
    for (ft_name, avg_ret, n_trades) in individual_performances
        println(@sprintf("  %s: %.3f bp (%d trades)", ft_name, avg_ret, n_trades))
    end
    
    # 상관관계 매트릭스 계산
    train_data = Matrix(df_evv_train[!, ft_names])
    valid_rows = .!any(isnan.(train_data), dims=2)[:]
    train_clean = train_data[valid_rows, :]
    corr_matrix = cor(train_clean)
    
    # 각 피처의 평균 절대 상관계수 계산
    avg_abs_corrs = []
    for (i, ft_name) in enumerate(ft_names)
        other_corrs = [abs(corr_matrix[i, j]) for j in 1:length(ft_names) if i != j]
        push!(avg_abs_corrs, (ft_name, mean(other_corrs)))
    end
    sort!(avg_abs_corrs, by=x->x[2])
    
    println("\n피처별 평균 절대 상관계수 (낮을수록 독립적):")
    for (ft_name, avg_corr) in avg_abs_corrs
        println(@sprintf("  %s: %.4f", ft_name, avg_corr))
    end
    
    # 성능-독립성 스코어 계산
    performance_dict = Dict(perf[1] => perf[2] for perf in individual_performances)
    independence_dict = Dict(corr[1] => 1.0 - corr[2] for corr in avg_abs_corrs)  # 1에서 빼서 독립성 점수로 변환
    
    combined_scores = []
    for ft_name in ft_names
        perf_score = performance_dict[ft_name]
        indep_score = independence_dict[ft_name]
        # 정규화 (성능은 평균을 중심으로, 독립성은 0-1 스케일)
        normalized_perf = perf_score / (abs(mean([p[2] for p in individual_performances])) + 1e-8)
        combined_score = 0.6 * normalized_perf + 0.4 * indep_score  # 성능 60%, 독립성 40% 가중치
        push!(combined_scores, (ft_name, combined_score, normalized_perf, indep_score))
    end
    
    sort!(combined_scores, by=x->x[2], rev=true)
    
    println("\n종합 점수 (성능 60% + 독립성 40%):")
    for (ft_name, combined, perf_norm, indep) in combined_scores
        println(@sprintf("  %s: %.4f (성능: %.4f, 독립성: %.4f)", 
                ft_name, combined, perf_norm, indep))
    end
    
    return individual_performances, avg_abs_corrs, combined_scores
end

end  # 함수 정의 블록 끝


begin  # 분석 실행 블록
println("\n🔍 피처 직교성 분석을 시작합니다...")
orthogonality_results = comprehensive_orthogonality_analysis(df_evv_train, df_evv_test, ft_names)

# 추가 분석들
println("\n📊 추가 분석을 진행합니다...")

# 클러스터링 분석
corr_train = orthogonality_results["correlation_matrices"][1]
similar_pairs = analyze_feature_clustering(corr_train, ft_names)

# 피처 선택 권장
recommended_idx, recommended_names = recommend_feature_selection(corr_train, 
                                                               orthogonality_results["pca_results"][1], 
                                                               ft_names)

# 중요도-직교성 트레이드오프 분석  
individual_perf, avg_corrs, combined_scores = analyze_importance_orthogonality_tradeoff(
    df_evv_train, df_evv_test, ft_names)

println("\n" * "="^80)
println("🎯 최종 분석 요약")
println("="^80)
println(@sprintf("전체 피처 수: %d개", length(ft_names)))
println(@sprintf("직교성 점수: %.4f", orthogonality_results["orthogonality_score"]))
println(@sprintf("PCA 분산 분산도: %.4f", orthogonality_results["pca_score"]))
println(@sprintf("권장 피처 수: %d개", length(recommended_names)))
println("\n권장 피처 목록:")
for ft_name in recommended_names
    println("  ✓ $ft_name")
end
end  # 분석 실행 블록 끝



