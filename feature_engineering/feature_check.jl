

include("./../const.jl")
include("./../utils.jl")
include("./../feature_generator/feature_generator.jl")

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


######################################



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


px = [1, 2, 3, 5, 10, 25, 50, 75, 90, 95, 99]
for (ft_name, ft_gen) in ft_gen_map
    plt = plot(title="$(ft_name)")
    for symbol in symbols
        df = df_train_map[symbol]
        ft = ft_gen(df)
        py_l, py_h = simple_view_feature_power_return(ft, df.ret_bp, is_display=false)
        plot!(plt, px, py_l, label="$(symbol)_low")
        plot!(plt, px, py_h, label="$(symbol)_high")
    end
    # display(plt)
    savefig(plt, "./fig/feature/$(ft_name).png")
end








