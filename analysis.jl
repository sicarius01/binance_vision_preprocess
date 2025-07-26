

include("const.jl")
include("utils.jl")



sb_dir = joinpath(bar_dir, symbol)
file_names = readdir(sb_dir)
ffn_list = [joinpath(sb_dir, file_name) for file_name in file_names]

df_list = [Parquet2.readfile(ffn) |> DataFrame for ffn in ffn_list]
df = vcat(df_list...)
nothing



tg_sec = 20
tg_interval = Int(round(tg_sec / interval_sec))

fut_c = [df.c[tg_interval + 1 : end]; fill(NaN, tg_interval)]
ret_bp = (fut_c .- df.c) .* 10_000 ./ df.c
ret_bp[isnan.(ret_bp)] .= 0.0
df.ret_bp = ret_bp


feature_1 = df.vwap .- df.c
feature_1[isnan.(feature_1)] .= 0.0
df.feature_1 = feature_1
histogram(feature_1)
simple_view_feature_power(feature_1, df.ret_bp)
simple_view_feature_power(feature_1, abs.(df.ret_bp))


feature_2 = [NaN; diff(df.c)]
feature_2[isnan.(feature_2)] .= 0.0
df.feature_2 = feature_2
histogram(feature_2)
simple_view_feature_power(feature_2, df.ret_bp)
simple_view_feature_power(feature_2, abs.(df.ret_bp))


























