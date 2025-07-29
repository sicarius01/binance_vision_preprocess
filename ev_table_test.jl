


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


#################################### calc ev table ####################################
X = ofi
ret_bp = df.ret_bp
th_vec = [(0.1:0.1:0.9); Float64.(1:99); (99.1:0.1:99.9)]
tb_info = get_ev_info(X, ret_bp, th_vec)

plot([(tbi[1] + tbi[2])/2.0 for tbi in tb_info], [tbi[3] for tbi in tb_info])
scatter([(tbi[1] + tbi[2])/2.0 for tbi in tb_info], [tbi[3] for tbi in tb_info], markersize=3)


# find_interval_index(tb_info, Inf)

wsv_vec = get_wsv_parallel(tb_info, X)


si = [i for i in 1:300:length(X)]
scatter(wsv_vec[si], df.ret_bp[si], markersize=2)
simple_view_feature_power(wsv_vec, df.ret_bp)


sl, ss = get_signals(wsv_vec, 0.5)
plt = backtest_sica(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, 15)
plt, tr_res_vec = backtest_sica_2(sl, ss, df.WAP_Lag_200ms, df.WAP_Lag_0ms, df.timestamp, is_display=true)




wap = df.WAP_Lag_0ms
avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec, wap)
begin
    plot(avg_norm_paths, label="avg")
    plot!(median_norm_paths, label="median")
end


period = [tr_res[2] - tr_res[1] for tr_res in tr_res_vec]
histogram(period ./ 60, xlims=(-5, 50))


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












# # 성능 비교 테스트
# println("=== 성능 비교 테스트 ===")
# println("데이터 크기: $(length(X))")
# println("CPU 쓰레드 수: $(Threads.nthreads())")

# # 작은 샘플로 먼저 테스트 (정확성 확인)
# sample_size = min(10000, length(X))
# X_sample = X[end-sample_size : end]
# println("\n1. 정확성 검증 (샘플 크기: $sample_size)")

# t1 = time()
# result_original = [get_wsv(tb_info, xi) for xi in X_sample]
# t2 = time()
# time_original = t2 - t1
# println("기존 방법: $(round(time_original, digits=3)) sec")

# t1 = time()
# result_fast = get_wsv_fast(tb_info, X_sample)
# t2 = time()
# time_fast = t2 - t1
# println("벡터화 버전: $(round(time_fast, digits=3)) sec")
# println("속도 향상: $(round(time_original/time_fast, digits=1))x")

# # 결과 비교 (NaN 처리)
# diff = abs.(result_original .- result_fast)
# valid_mask = .!isnan.(result_original) .& .!isnan.(result_fast)
# max_diff = maximum(diff[valid_mask])
# println("최대 차이: $(round(max_diff, digits=8))")

# if max_diff < 1e-10
#     println("✅ 정확성 검증 통과!")
    
#     println("\n2. 전체 데이터 성능 테스트")
    
#     # 기존 방법 (참고용 - 너무 느리면 스킵)
#     if length(X) <= 100000
#         t1 = time()
#         wsv_vec_original = [get_wsv(tb_info, xi) for xi in X]
#         t2 = time()
#         println("기존 방법: $(round(t2 - t1, digits=3)) sec")
#     else
#         println("기존 방법: 스킵 (데이터가 너무 큼)")
#     end
    
#     # 벡터화 버전
#     t1 = time()
#     wsv_vec_fast = get_wsv_fast(tb_info, X)
#     t2 = time()
#     time_fast_full = t2 - t1
#     println("벡터화 버전: $(round(time_fast_full, digits=3)) sec")
    
#     # 병렬 처리 버전
#     t1 = time()
#     wsv_vec_parallel = get_wsv_parallel(tb_info, X)
#     t2 = time()
#     time_parallel = t2 - t1
#     println("병렬 처리 버전: $(round(time_parallel, digits=3)) sec")
        
#     println("\n3. 속도 향상 요약")
#     base_time = time_fast_full
#     println("벡터화 대비:")
#     println("  - 병렬 처리: $(round(base_time/time_parallel, digits=1))x")
    
#     # 최고 성능 버전 선택
#     times = [time_fast_full, time_parallel]
#     namess = ["벡터화", "병렬"]
#     best_idx = argmin(times)
#     println("\n🏆 최고 성능: $(namess[best_idx]) ($(round(times[best_idx], digits=3)) sec)")
    
#     # 최고 성능 버전을 wsv_vec에 할당
#     if best_idx == 1
#         wsv_vec = wsv_vec_fast
#     elseif best_idx == 2
#         wsv_vec = wsv_vec_parallel
#     elseif best_idx == 3  
#         wsv_vec = wsv_vec_cache
#     elseif best_idx == 4
#         wsv_vec = wsv_vec_lookup
#     else
#         wsv_vec = wsv_vec_ultra
#     end
    
# else
#     println("❌ 정확성 검증 실패 - 기존 방법 사용")
#     t1 = time()
#     wsv_vec = [get_wsv(tb_info, xi) for xi in X]
#     t2 = time()
#     println("기존 방법 처리 시간: $(round(t2 - t1, digits=3)) sec")
# end



# # 기존 방식
# t1 = time()
# result_original = [get_wsv(tb_info, xi) for xi in X]
# t2 = time()
# time_original = t2 - t1
# println("기존 방법: $(round(time_original, digits=3)) sec")

# # 벡터화 버전
# t1 = time()
# wsv_vec_fast = get_wsv_fast(tb_info, X)
# t2 = time()
# time_fast_full = t2 - t1
# println("벡터화 버전: $(round(time_fast_full, digits=3)) sec")

# # 병렬 처리 버전
# t1 = time()
# wsv_vec_parallel = get_wsv_parallel(tb_info, X)
# t2 = time()
# time_parallel = t2 - t1
# println("병렬 처리 버전: $(round(time_parallel, digits=3)) sec")



# all(wsv_vec[.!isnan.(wsv_vec)] .== result_original[.!isnan.(result_original)])
# all(wsv_vec[.!isnan.(wsv_vec)] .== wsv_vec_fast[.!isnan.(wsv_vec_fast)])
# all(wsv_vec[.!isnan.(wsv_vec)] .== wsv_vec_parallel[.!isnan.(wsv_vec_parallel)])